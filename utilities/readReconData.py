import numpy as np
import sigpy as sp
import os


def load_data(folder_path, device = sp.cpu_device, nCoilUse = -1):
    
    img_shape = []
    
    for fname in os.listdir(folder_path):
        if '_BefGrid' in fname:
            ksp = np.load(folder_path + fname)
            ksp = ksp.squeeze()
            if(nCoilUse > 0):
                ksp = ksp[0:nCoilUse,...]
        elif '_SDC' in fname:
            dcf = np.load(folder_path + fname)
            dcf = dcf.squeeze()
        elif '_CSM' in fname:
            csm = np.load(folder_path + fname)
            csm = csm.squeeze()
            csm = np.moveaxis(csm, [-1, -2, -3], [-3, -2, -1])
            csm = csm[:, ::-1,::-1,::-1]
            B1_mag = np.sum(np.abs(csm * np.conj(csm)), axis=0)
            B1_mag = np.sqrt(np.clip(B1_mag, a_min=1e-5, a_max=None))            
            csm = csm / B1_mag
            
            if(nCoilUse > 0):
                csm = csm[0:nCoilUse,...]
                
            img_shape = csm.shape[1:]
            
        elif '_CFM' in fname:
            cfm = np.load(folder_path + fname)
            cfm = cfm.squeeze()
            cfm = np.moveaxis(cfm, [-1, -2, -3], [-3, -2, -1])
            cfm = cfm[:, ::-1,::-1,::-1]
        elif '_Coordinates' in fname:
            coord = np.load(folder_path + fname)
            coord = coord.squeeze()    
            
        elif '_REG' in fname:
            reg = np.load(folder_path + fname)
            reg = reg.squeeze()
            reg = np.moveaxis(reg, [-1, -2, -3], [-3, -2, -1])
            reg = reg[::-1,::-1,::-1]
            reg = np.abs(reg)
            reg = 1 / np.clip(reg, a_min=1e-1, a_max=None)
            #reg = np.clip(reg, a_min=1e-1, a_max=None)

     
   
    return ksp, dcf, csm,cfm, coord, reg, img_shape