import functools
import os

import numpy as np
from mayavi import mlab
import pylab as plt
import time
from PointcloudVoxelizer.source import pointclouds_to_voxelgrid
from frame_diff_algo import get_imu_data, rotate
from funtiontest import cmp, load_velo_scan
from PIL import Image


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud


def filter_ground(PointCloud, Width, Height, grid_size=10):
    """
    过滤地面
    :param PointCloud: 点云
    :param Width: 图像宽度
    :param Height: 图像高低
    :param grid_size: 网格粒度
    :return: 过滤地面后的点云
    """
    start = time.clock()
    indices = []
    # 划分网格
    for i in range(0, Width, grid_size):
        for j in range(0, Height, grid_size):
            ids = np.where(
                (i <= PointCloud[:, 0]) & (PointCloud[:, 0] < i + grid_size) &
                (j <= PointCloud[:, 1]) & (PointCloud[:, 1] < j + grid_size)
            )
            if ids[0].shape[0] > 0:
                if np.max(PointCloud[ids][:, 2]) > 2.5:
                    indices.append(ids)
    indices = np.hstack(indices)
    PointCloud = np.squeeze(PointCloud[indices])
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    return PointCloud


def makeBVFeature(PointCloud_, BoundaryCond, Discretization):  # Dis=40/512
    # 1024 x 1024 x 3
    Height = Discretization + 1  # 1024 + 1
    Width = Discretization + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(
        PointCloud[:, 0] / abs(BoundaryCond['maxX'] - BoundaryCond['minX']) * Discretization
    ))  # x  倍增
    PointCloud[:, 1] = np.int_(np.floor(
        PointCloud[:, 1] / abs(BoundaryCond['maxY'] - BoundaryCond['minY']) * Discretization
    ))  # y, 倍增+平移
    # ) + Width / 2)  # y, 倍增+平移
    # 过滤地面
    # PointCloud = filter_ground(PointCloud, Width=Width, Height=Height, grid_size=40)
    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))  # 空 矩阵

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)  # 去重
    PointCloud_frac = PointCloud[indices]  # 剩余点
    # some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]  # 高度作为数值

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]  # 反射率

    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts  # 用的是出现的次数作为对应坐标的数值
    # 对三个通道的值进行归一化
    densityMap = densityMap / densityMap.max()
    heightMap = heightMap / heightMap.max()
    intensityMap = intensityMap / intensityMap.max()
    # plt.subplot(1, 3, 1)
    # plt.imshow(densityMap[:, :])
    # plt.xlabel('densityMap')
    # plt.subplot(1, 3, 2)
    # plt.imshow(heightMap[:, :])
    # plt.xlabel('heightMap')
    # plt.subplot(1, 3, 3)
    # plt.imshow(intensityMap[:, :])
    # plt.xlabel('intensityMap')
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()
    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap  # r_map
    RGB_Map[:, :, 1] = heightMap  # g_map
    RGB_Map[:, :, 2] = intensityMap  # b_map

    save = np.zeros((512, 1024, 3))
    save = RGB_Map[0:512, 0:1024, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return save


if __name__ == '__main__':

    border_thresh = (-50, -60, -1.9, 50, 60, 2.4)
    boundary = {'minX': -50, 'maxX': 50, 'minY': -20, 'maxY': 60, 'minZ': -1.9, 'maxZ': 2.4, }
    for seq_id in range(3, 4):
        # seq_id = 2
        # 初始化imu数据
        imu_path = './data/imuseq/%d' % seq_id
        # imu数据处理器
        imu_data_processer = get_imu_data()
        next(imu_data_processer)
        velo_path = './data/veloseq/%d' % seq_id
        velo_list = []
        # 计算多帧的聚类结果
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cmp)):
            frame_id = int(frame_file.replace('.bin', ''))
            print(frame_id)
            # 提取点云
            velo = load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))
            # 消除多余点
            velo = removePoints(velo, boundary)
            # 平移velo 将boundary的最左下角作为坐标原点，防止转换为feature map时出现负值
            velo[:, 0] += abs(boundary['minX'])
            velo[:, 1] += abs(boundary['minY'])
            velo[:, 2] += abs(boundary['minZ'])
            # mlab.points3d(velo[:, 0], velo[:, 1], velo[:, 2], velo[:, 2], mode="point", )
            # mlab.show()
            # 转转换为平面图像
            feature_map = makeBVFeature(velo, boundary, 512)
            # Image.fromarray(np.asarray(feature_map, dtype=np.float32)).save('./output/%d.png' % frame_id)
            plt.imsave('./output/%d.png' % frame_id, feature_map)
