import sigpy as sp

class CGSENSE(object):
    def __init__(self, ksp, coord, dcf, csm, reg, tol=1e-6, labmda = 0.05, max_iter=100,
                 device=sp.cpu_device):
        
        # Copy Data to GPU         

        self.device = sp.Device(device)
        self.xp = device.xp
        
       
        self.csm = sp.to_device(csm, device=device)
       
        self.reg = sp.to_device(reg, device=device)
        
        self.tol = tol
        self.max_iter = max_iter
       
        self.ksp = sp.to_device(ksp, device=device)
        
        # Prepare Coordinates for Gridding    
        img_shape = csm[1:].shape

        coord[...,0] *= img_shape[-1] 
        coord[...,1] *= img_shape[-2]
        coord[...,2] *= img_shape[-3]
        g = sp.estimate_shape(coord)
        coord[...,0] *= (img_shape[-1] / g[0])
        coord[...,1] *= (img_shape[-2] / g[1])
        coord[...,2] *= (img_shape[-3] / g[2]) 
        
        g = sp.estimate_shape(coord)
        print(g)
        
        self.coord = sp.to_device(coord, device=device)
        self.dcf = sp.to_device(dcf, device=device)
        self.labmda = labmda
        
        self.show_pbar = True

        self._firstGuess()
        


    def _firstGuess(self):

        with self.device:
           
            mrimg_adj = sp.nufft_adjoint(self.ksp * self.dcf, self.coord)            
            self.mrimg_adj = self.xp.sum(mrimg_adj *  self.xp.conj(self.csm), axis= 0)            
            #pl.ImagePlot(self.mrimg_adj[:,:,100], title='GPU Gridding')

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
            
            Ad = sp.nufft_adjoint(self.dcf * sp.nufft(d * self.csm, self.coord), self.coord) 
            Ad = self.xp.sum(Ad*self.xp.conj(self.csm), axis=0)
            
            # regularization
            Ad += self.labmda * (1 + self.reg) * d
            
            # CG
            d_last, r_last, x_last, rHr = self._do_cg(d, r, x, Ad)
            
            if i == 1:
                rHr0 = rHr
                
            cg_res = self.xp.abs(rHr / rHr0)
       
        return sp.to_device(self.mrimg_adj, device=sp.Device(-1)), sp.to_device(x_last, device=sp.Device(-1))