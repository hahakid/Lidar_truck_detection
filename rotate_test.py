import functools
import os
import csv

import mayavi
import pyproj
import numpy as np
import pylab as plt
from funtiontest import load_velo_scan_v0
from mayavi import mlab

# 坐标对齐
velo_path0 = './data/Data-2019-01-21-23-28-32/VLP160'
velo_path1 = './data/Data-2019-01-21-23-28-32/VLP161'
steps = []
for i in range(80, 81):
    velo0 = load_velo_scan_v0(os.path.join(velo_path0, '%d.bin' % i))
    velo1 = load_velo_scan_v0(os.path.join(velo_path1, '%d.bin' % i))
    # velo = np.vstack((velo0, velo1))[:150000, :]
    velo = np.vstack((velo0, velo1))[:, :3]
    steps.append(np.ones([velo.shape[0], 1]) * (0))

    # 旋转
    theta = -30 / 180 * np.pi
    R_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    velo_rotated = np.matmul(R_matrix, velo.T).T
    steps.append(np.ones([velo_rotated.shape[0], 1]) * (90))
    velo = np.vstack([velo, velo_rotated])
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(velo[:, 0], velo[:, 1], velo[:, 2],
                         np.vstack(steps).reshape(-1),  # Values used for Color
                         mode="point",
                         colormap='cool',  # 'bone', 'copper', 'gnuplot'
                         figure=fig,
                         # color=(0, 1, 0),  # Used a fixed (r,g,b) instead
                         scale_factor=2,
                         )
    mayavi.mlab.show()
