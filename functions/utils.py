import functools
import numpy as np
import os
from functions import top_view
import matplotlib.pyplot as plt
import struct
from mayavi import mlab
import mayavi
from sklearn.cluster import DBSCAN

border_thresh = (-30, -20, -1.6, 30, 40, 2.4)


def cmp(a, b):
    l, r = int(a.replace('.bin', '')), int(b.replace('.bin', ''))
    return -1 if l < r else 1 if l > r else 0


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def load_velo_scan_v0(velo_filename):
    bin_file = open(velo_filename, 'rb')
    # 读取个数
    count = int.from_bytes(bin_file.read(4), byteorder='little')
    # 提取x y z
    x = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    y = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    z = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    reflect = [struct.unpack('B', bin_file.read(1))[0] for _ in range(count)]
    rgb_filler = [0 for _ in range(len(x))]
    scan = np.array([x, y, z, rgb_filler, rgb_filler, rgb_filler, reflect]).squeeze().T
    return scan


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan
