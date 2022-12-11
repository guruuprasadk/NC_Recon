# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:19:39 2022

@author: Guru Krishnamoorthy
"""
# Import all the necessary functions
#import numpy as np
#import matplotlib.pyplot as plt
import sys
import sigpy as sp
import matplotlib.pyplot as plt
import numpy as np
import sigpy.mri as mr

sys.path.append('C:/Users/philipsmr/Python Scripts/Repo/python/utilities')
from readReconData import load_data

#from imageViewer import IndexTracker
from CGSENSE import CGSENSE

from imageViewer import ImageViewer3D

    # Read npy data
device = sp.Device(0)
folder_path = 'D:/DataSet/FLORET_3HUB_R4/'
ksp, dcf, cfm, coord, reg, imageShape = load_data(folder_path, device, nCoilUse = 4)
# %%

us1 = np.arange(np.round(ksp.shape[1] * 0.7).astype(np.int32))
us2 = np.arange(np.round(ksp.shape[2] * 0.7).astype(np.int32))

np.random.shuffle(us1)
np.random.shuffle(us2)

kspU = ksp[:,us1,:,:]
dcfU = dcf[us1,:,:]
coordU = coord[us1,:,:,:]

kspU = kspU[:,:,us2,:]
dcfU = dcfU[:,us2,:]
coordU = coordU[:,us2,:,:]

# %%
grid_img, cg_img, cg_res  = CGSENSE(ksp, coord, dcf, cfm, reg, tol = 1e-5, labmda = 0.05, max_iter=100, device=device).run()

  # %%      
fig, ax = plt.subplots(1, 1)
grid_imgM = np.moveaxis(grid_img, [-1, -2, -3], [-3, -2, -1])
tracker = ImageViewer3D(ax, grid_imgM, 'Grid')
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
# %%
fig, ax = plt.subplots(1, 1)
cg_imgM = np.moveaxis(cg_img, [-1, -2, -3], [-3, -2, -1])
tracker = ImageViewer3D(ax, cg_imgM, 'cg_img')
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
# %%
np.save(folder_path + 'grid_img', grid_img)
np.save(folder_path + 'cg_img_NoReg', cg_img)
