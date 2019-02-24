import csv
import functools
import os
import pyproj
import numpy as np
import matplotlib as mpl

import pylab as plt


def cmp_file(a: str, b: str):
    """
    文件名的纯数字排序比较函数
    :param a:
    :param b:
    :return:
    """
    file_number_a = int(a[:a.rfind('.')])
    file_number_b = int(b[:b.rfind('.')])
    if file_number_a > file_number_b:
        return 1
    else:
        return -1


seq_id = 1
imu_root = '../data/imuseq/%d' % seq_id
origin = None
path_points = []
for imu_file_path in sorted(os.listdir(imu_root), key=functools.cmp_to_key(cmp_file)):
    f_csv = csv.reader(open(os.path.join(imu_root, imu_file_path), 'r'), delimiter=' ')
    lat, lon, direction, v, rtk = next(f_csv)
    # 坐标系转换
    p1 = pyproj.Proj(init="epsg:4610")  # 定义数据地理坐标系
    p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
    x1, y1 = p1(float(lon), float(lat))
    x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    if origin is None:
        origin = (x2, y2)
    # 调整以第一个点坐标为原点
    x2 -= origin[0]
    y2 -= origin[1]
    path_points.append([x2, y2, float(v)])
    print(float(direction))
path_points = np.array(path_points)
plt.scatter(path_points[:, 0], path_points[:, 1], s=0.5, c=path_points[:, 2], cmap=mpl.cm.hot_r)
plt.colorbar()
plt.show()
