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

sys.path.append('C:/Users/philipsmr/Python Scripts/Repo/python/utilities')
from readReconData import load_data

#from imageViewer import IndexTracker
from CGSENSE import CGSENSE

from imageViewer import ImageViewer3D

# Read npy data
device = sp.Device(0)
folder_path = 'D:/DataSet/FLORET_3HUB_R4/'
ksp, dcf, cfm, coord, reg, imageShape = load_data(folder_path, device)

# %%
grid_img, cg_img  = CGSENSE(ksp, coord, dcf, cfm, reg, tol = 1e-5, labmda = 0.05, max_iter=100, device=device).run()
# %%
fig, ax = plt.subplots(1, 1)
tracker = ImageViewer3D(ax, grid_img, 'Grid')
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
# %%
fig, ax = plt.subplots(1, 1)
tracker = ImageViewer3D(ax, cg_img, 'cg_img')
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()
# %%
np.save(folder_path + 'grid_img', grid_img)
np.save(folder_path + 'cg_img', cg_img)
# %%
