import functools
import os
import numpy as np
import struct
from mayavi import mlab
import pylab as plt
import time
#from PointcloudVoxelizer.source import pointclouds_to_voxelgrid
#from coordinate import get_imu_data, rotate
#from funtiontest import cmp, load_velo_scan
#from PIL import Image
import top_view
import top_view1
import pyproj
import csv

# read from combined bin file with all element float32
def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

# read from raw program output,the format is not unique. And need further combine opt
def load_velo_scan_raw(velo_filename):
    bin_file = open(velo_filename, 'rb')
    # 读取个数
    count = int.from_bytes(bin_file.read(4), byteorder='little')
    # 提取x y z
    x = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    y = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    z = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    reflect = [struct.unpack('B', bin_file.read(1))[0] for _ in range(count)]
    #rgb_filler = [0 for _ in range(len(x))]
    #scan = np.array([x, y, z, rgb_filler, rgb_filler, rgb_filler, reflect]).squeeze().T
    scan = np.array([x, y, z, reflect]).squeeze().T
    return scan

def cmp(a, b):
    l, r = int(a.replace('.bin', '')), int(b.replace('.bin', ''))
    return -1 if l < r else 1 if l > r else 0

#distance filter
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

def scale_to_255(a, min, max, dtype=np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def singlechannelfeature(PointCloud_, BoundaryCond, Discretization):
    Height = Discretization   #480
    Width = Discretization
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(
        PointCloud[:, 0] / abs(BoundaryCond['maxX'] - BoundaryCond['minX']) * Discretization
    ))  # x  倍增
    PointCloud[:, 1] = np.int_(np.floor(
        PointCloud[:, 1] / abs(BoundaryCond['maxY'] - BoundaryCond['minY']) * Discretization
    ))
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))

    pixel_values = np.clip(a=PointCloud[:, 2],
                           a_min=BoundaryCond['minZ'],
                           a_max=BoundaryCond['maxZ'])

    pixel_values = scale_to_255(pixel_values,
                                min=BoundaryCond['minZ'],
                                max=BoundaryCond['maxZ'])
    im = np.zeros([Height, Width], dtype=np.uint8)
    im[Height, Width] = pixel_values
    return im

# pc, , 512
def makeBVFeature(PointCloud_, BoundaryCond, Discretization):  # Dis=
    Height = Discretization + 1  #
    Width = Discretization + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / abs(BoundaryCond['maxX'] - BoundaryCond['minX']) * Discretization))  # x  倍增
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / abs(BoundaryCond['maxY'] - BoundaryCond['minY']) * Discretization))  # y, 倍增+平移

    indices = np.lexsort((PointCloud[:, 1], PointCloud[:, 2], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map & Intensity Map & DensityMap
    heightMap = np.zeros((Height, Width))  # 空 矩阵
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    #_, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)  # 去重
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)

    PointCloud_remain = PointCloud[indices]  # 剩余点
    # some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_remain[:, 0]), np.int_(PointCloud_remain[:, 1])] = PointCloud_remain[:, 2]  # 高度作为数值

    #PointCloud_top = PointCloud[indices]
    intensityMap[np.int_(PointCloud_remain[:, 0]), np.int_(PointCloud_remain[:, 1])] = PointCloud_remain[:, 3]  # 反射率
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    densityMap[np.int_(PointCloud_remain[:, 0]), np.int_(PointCloud_remain[:, 1])] = normalizedCounts  # 用的是出现的次数作为对应坐标的数值

    # 对三个通道的值进行归一化,后期输出的话可能要×255
    densityMap = densityMap / densityMap.max()
    heightMap = heightMap / heightMap.max()
    intensityMap = intensityMap / intensityMap.max()

    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap  # r_map
    RGB_Map[:, :, 1] = heightMap  # g_map
    RGB_Map[:, :, 2] = intensityMap  # b_map

    save = np.zeros((Discretization, Discretization, 3))
    save = RGB_Map[0:Discretization, 0:Discretization, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return save

#雷达文件合并
def singleframe():
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.6, 'maxZ': 2.4, }
    pixel=480 # pixel 480=32*15(yolo)=60*8
    for seq_id in range(1, 2):  # 1-30
        # seq_id = 2
        # 初始化imu数据
        father_path = 'F:/daxie/daxie/data/frist/'
        save_path_3c = 'F:/daxie/daxie/data/frist/birdview/feature_3c/%d' % seq_id
        save_path_1c = 'F:/daxie/daxie/data/frist/birdview/feature_1c/%d' % seq_id

        if not os.path.exists(save_path_3c):
            os.makedirs(save_path_3c)
        if not os.path.exists(save_path_1c):
            os.makedirs(save_path_1c)

        velo_path = os.path.join(father_path, 'velo/%d' % seq_id)
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

            feature_map_1c = singlechannelfeature(velo,boundary,pixel)  # top_view1.point_cloud_2_birdseye(velo, res=0.1, side_range=(-30, 30),
            plt.imsave(save_path_1c + '/%d.png' % frame_id, feature_map_1c)

            # 转转换为平面图像
            #feature_map_3c = makeBVFeature(velo, boundary, pixel)
            #plt.imsave(save_path_3c + '/%d.png' % frame_id, feature_map_3c)

#雷达文件分开
def singleframe2():
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.6, 'maxZ': 2.4, }
    pixel=480 # pixel 480=32*15(yolo)=60*8
    for seq_id in range(1, 2):  # 1-30
        # seq_id = 2
        # 初始化imu数据
        father_path = 'F:/daxie/daxie/data/frist/'
        save_path_3c = 'F:/daxie/daxie/data/frist/birdview/feature_3c/%d' % seq_id
        save_path_1c = 'F:/daxie/daxie/data/frist/birdview/feature_1c/%d' % seq_id

        if not os.path.exists(save_path_3c):
            os.makedirs(save_path_3c)
        if not os.path.exists(save_path_1c):
            os.makedirs(save_path_1c)

        velo_path = os.path.join(father_path, 'velo/%d' % seq_id)
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

            feature_map_1c = singlechannelfeature(velo,boundary,pixel)  # top_view1.point_cloud_2_birdseye(velo, res=0.1, side_range=(-30, 30),
            plt.imsave(save_path_1c + '/%d.png' % frame_id, feature_map_1c)

            # 转转换为平面图像
            #feature_map_3c = makeBVFeature(velo, boundary, pixel)
            #plt.imsave(save_path_3c + '/%d.png' % frame_id, feature_map_3c)

if __name__ == '__main__':
    #singleframe()
    singleframe2()

    #multiframe
    '''
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.6, 'maxZ': 2.4, }
    #path = '/media/kid/workspace/daxie/output/birdview'

    for seq_id in range(1, 2):  # 1-30
        # seq_id = 2
        # 初始化imu数据

        father_path = 'F:/daxie/daxie/data/frist/'
        save_path_3c = 'F:/daxie/daxie/data/frist/birdview/feature_3c/%d' % seq_id
        save_path_1c = 'F:/daxie/daxie/data/frist/birdview/feature_1c/%d' % seq_id

        if not os.path.exists(save_path_3c):
            os.makedirs(save_path_3c)
        if not os.path.exists(save_path_1c):
            os.makedirs(save_path_1c)

        velo_path = os.path.join(father_path, 'velo/%d' % seq_id)
        print(velo_path)
        velo_list = sorted(os.listdir(velo_path))
        print(velo_list)
        # 计算多帧的聚类结果
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cmp)):
            frame_id = int(frame_file.replace('.bin', ''))
            if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + 1))):
                break

            print(frame_id)

            #velo, _ = get_window_merged(start_pos=frame_id, length=2)
            velo1=load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))
            velo2 = load_velo_scan(os.path.join(velo_path, '%d.bin' % (frame_id+1)))
            velo=np.vstack((velo1,velo2))
            # 提取点云
            # 消除多余点
            feature_map_1c = top_view.point_cloud_2_birdseye(velo, res=480, side_range=(-30, 30),
                                                             fwd_range=(-10, 50), height_range=(-1.6, 2.4))
            plt.imsave(save_path_1c + '/%d.png' % frame_id, feature_map_1c)

            velo = removePoints(velo, boundary)

            # 平移velo 将boundary的最左下角作为坐标原点，防止转换为feature map时出现负值
            velo[:, 0] += abs(boundary['minX'])
            velo[:, 1] += abs(boundary['minY'])
            velo[:, 2] += abs(boundary['minZ'])

            # 转转换为平面图像
            feature_map_3c = makeBVFeature(velo, boundary, 480)
            plt.imsave(save_path_3c + '/%d.png' % frame_id, feature_map_3c)
    '''

