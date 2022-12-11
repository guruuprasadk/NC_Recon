import numpy as np
import sigpy as sp
import os


def load_data(folder_path, device = sp.cpu_device, nCoilUse = -1):
    
    
    for fname in os.listdir(folder_path):
        if '_BefGrid' in fname:
            ksp = np.load(folder_path + fname)
            ksp = ksp.squeeze()
            if(nCoilUse > 0):
                ksp = ksp[0:nCoilUse,...]
        elif '_SDC' in fname:
            dcf = np.load(folder_path + fname)
            dcf = dcf.squeeze()
        elif '_CFM' in fname:
            cfm = np.load(folder_path + fname)
            cfm = cfm.squeeze()
            
            if(nCoilUse > 0):
                cfm = cfm[0:nCoilUse,...]
            
        elif '_Coordinates' in fname:
            coord = np.load(folder_path + fname)
            coord = coord.squeeze()    
            
        elif '_REG' in fname:
            reg = np.load(folder_path + fname)
            reg = reg.squeeze()
            reg = np.abs(reg)
            
    img_shape =  cfm.shape[1:]
           
    if(coord.shape[-1] == 2):
        dcf = np.reshape(dcf,(coord.shape[0], coord.shape[1]))
        
              
    return ksp, dcf, cfm, coord, reg, img_shape