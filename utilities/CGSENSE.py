import sigpy as sp
import numpy as np
from scipy.signal import tukey
import sigpy.plot as pl

class CGSENSE(object):
    def __init__(self, ksp, coord, dcf, cfm, reg, tol=1e-6, labmda = 0.05, max_iter=100,
                 device=sp.cpu_device):
        
        # Copy Data to GPU       
        
        self.device = sp.Device(device)
        self.xp = device.xp
        
        #print(cfm.shape)
        
        nonCart3D = (coord.shape[-1] == 3)        
        if(nonCart3D):
            cfm = np.moveaxis(cfm, [-1, -2, -3], [-3, -2, -1])
            cfm = cfm[:, ::-1,::-1,::-1]
            reg = np.moveaxis(reg, [-1, -2, -3], [-3, -2, -1])
            reg = reg[::-1,::-1,::-1]
        else:
            cfm = np.moveaxis(cfm, [-1, -2, -3], [-2, -1, -3])
            cfm = cfm[:, :,::-1,::-1]
            reg = np.moveaxis(reg, [-1, -2, -3], [-2, -1, -3])
            reg = reg[:,::-1,::-1]
        
        #np.save('D:/DataSet/SSC_UTE/'  + 'Rotated_reg_img', reg)
        #print(cfm.shape)
        
        # Intensity normalization
        B1_mag = np.sum(np.abs(cfm * np.conj(cfm)), axis=0)
        B1_mag = np.sqrt(np.clip(B1_mag, a_min=1e-5, a_max=None))            
        cfm = np.conj(cfm) / B1_mag
        
        self.I = sp.to_device(np.clip(B1_mag, a_min=1e-5, a_max=None), device=device)
        
        self.cfm = sp.to_device(cfm, device=device)
       
        reg = 1 / np.clip(reg, a_min=1e-1, a_max=None)
        self.reg = sp.to_device(reg, device=device)
        
        self.tol = tol
        self.max_iter = max_iter
       
        self.ksp = sp.to_device(ksp, device=device)
        
        # Prepare Coordinates for Gridding    
        img_shape = cfm.shape[1:]
        
        if(nonCart3D):
            coord[...,0] *= img_shape[-1] 
            coord[...,1] *= img_shape[-2]
            coord[...,2] *= img_shape[-3]
            g = sp.estimate_shape(coord)
            coord[...,0] *= (img_shape[-1] / g[0])
            coord[...,1] *= (img_shape[-2] / g[1])
            coord[...,2] *= (img_shape[-3] / g[2]) 
            
            self.coord = sp.to_device(coord, device=device)
            self.dcf = sp.to_device(dcf, device=device)
        else:
            coord[...,0] *= img_shape[-1]
            coord[...,1] *= img_shape[-2]
            g = sp.estimate_shape(coord)
            coord[...,0] *= (img_shape[-1] / g[0])
            coord[...,1] *= (img_shape[-2] / g[1])
            
            self.coord = sp.to_device(coord, device=device)
            
            dcf = np.reshape(dcf,(coord.shape[0], coord.shape[1]))
            self.dcf = sp.to_device(dcf, device=device)
        
        self.img_shape = img_shape

        self.labmda = labmda

        self._firstGuess()
        


    def _firstGuess(self):

        with self.device:           
            mrimg_adj = sp.nufft_adjoint(self.ksp * self.dcf, self.coord, oversamp = 2.0)
            #np.save('D:/DataSet/SSC_UTE/'  + 'Grid_img', mrimg_adj)
           
            self.mrimg_adj = self.xp.sum(mrimg_adj *  self.xp.conj(self.cfm), axis= 0)  
           # pl.ImagePlot(self.mrimg_adj, z=0, title='Sensitivity Maps Estimated by ESPIRiT')

    def _do_cg(self, d_in, r_in, x_in, Ad_in):
            # Calculate alpha
            # r^H r / (d^H Ad)
            rHr = self.xp.dot(self.xp.conj(r_in.flatten()), r_in.flatten())
            dHAd = self.xp.dot(self.xp.conj(d_in.flatten()), Ad_in.flatten())
            alpha = rHr / dHAd
        
            # Calculate x(i+1)
            # x(i) + alpha d(i)
            x_out = x_in + alpha * d_in
        
            # Calculate r(i+1)
            # r(i) - alpha Ad(i)
            r_out = r_in - alpha * Ad_in
        
            # Calculate beta
            # r(i+1)^H r(i+1) / (r(i)^H r(i))
            r1Hr1 = self.xp.dot(self.xp.conj(r_out.flatten()), r_out.flatten())
            beta = r1Hr1 / rHr
        
            # Calculate d(i+1)
            # r(i+1) + beta d(i)
            d_out = r_out + beta * d_in
        
            return (d_out, r_out, x_out, rHr)

    def run(self):
        # initialize these here in case maxiter = 0
        d_last = None
        r_last = None
        x_last = self.mrimg_adj
    
        # CG iteration loop
        i = 0  # iteration counter
        cg_res = 1  # conjugate gradient residual (ratio with initial residual)
        cg_res_arr = np.zeros(self.max_iter, dtype=float,)
        while i < self.max_iter and cg_res > self.tol:
            if i == 0:
                # first iteration, use CLEAR combination
                r = self.mrimg_adj
                d = r.copy()
                x = self.xp.zeros_like(d)
            else:
                # for subsequent iterations, input the result of the last iter
                d = d_last
                r = r_last
                x = x_last

            i += 1
            print("\tSENSE Iteration: ", i, " cg_res: ", cg_res)
            
            
            
            Ad = sp.nufft_adjoint(self.dcf * sp.nufft(d * self.cfm, self.coord), self.coord, oversamp = 2.0) 
            Ad = self.xp.sum(Ad*self.xp.conj(self.cfm), axis=0)
            
            # regularization
            Ad += self.labmda * (1 + self.reg) * d

            
            # CG
            d_last, r_last, x_last, rHr = self._do_cg(d, r, x, Ad)
            
            if i == 1:
                rHr0 = rHr
                
            cg_res = self.xp.abs(rHr / rHr0)
            
            cg_res_arr[i] = cg_res

        grid_img = sp.to_device((self.mrimg_adj * self.I), device=sp.Device(-1))
        cg_img = sp.to_device((x_last * self.I), device=sp.Device(-1)) 
        
        # final filtering of x, clamps noisy samples extrapolated outside traj
        X = sp.fft(cg_img, axes=(-3, -2, -1))
        filter_n = 512
        w = tukey(2*filter_n, 0.15)
        x = y = z = np.linspace(-1, 1, self.img_shape[0])
        X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z)
        RR = np.sqrt(X_grid ** 2 + Y_grid ** 2 + Z_grid ** 2)
        RR *= filter_n
        RR[RR >= filter_n] = filter_n
        W = w[RR.astype(np.int32) + filter_n - 1]
        X *= W
        
        cg_img = sp.ifft(X, axes=(-3, -2, -1))
        
        return grid_img, cg_img, cg_res_arr[1:i]