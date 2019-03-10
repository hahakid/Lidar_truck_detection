# -*- encoding: utf-8 -*-

import functools
import os

import numpy as np
# from mayavi import mlab
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
#from PointcloudVoxelizer.source import pointclouds_to_voxelgrid
# from coordinate import get_imu_data, rotate
# from funtiontest import cmp, load_velo_scan
# from PIL import Image
# import top_view
import feature_projection
import pyproj
import csv
#import combine_pointCloud
import cv2

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

'''
def removePoints1(points, BoundaryCond):

    x_points = points[:, 0]
    y_points = points[:, 1]

    side_range = (BoundaryCond['minX'], BoundaryCond['maxX'])
    fwd_range = (BoundaryCond['minY'], BoundaryCond['maxY'])

    f_filt = np.logical_and((x_points > side_range[0]), (x_points < side_range[1]))
    s_filt = np.logical_and((y_points > fwd_range[0]), (y_points < fwd_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    PointCloud=points[indices]
    return PointCloud
'''

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def cmp(a, b):
    l, r = int(a.replace('.bin', '')), int(b.replace('.bin', ''))
    return -1 if l < r else 1 if l > r else 0

def point_cloud_2_birdseye(points,
                           res=512,
                           side_range=(-10., 10.),  # left-most to right-most
                           fwd_range = (-10., 10.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    # 提取每个轴的点数
    x_points = points[:, 1]
    y_points = points[:, 0]
    z_points = points[:, 2]

    # 过滤器 - 仅返回立方体内点的索引
    # 三个过滤器用于前后，左右，高度范围
    # 雷达坐标系中左侧是正Y轴
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    # indices = np.max(indices) - indices

    # KEEPERS 保留的点
    x_points = x_points[indices]+abs(fwd_range[0])
    y_points = y_points[indices]+abs(side_range[0])
    z_points = z_points[indices] + abs(height_range[0])

    # 转换为像素位置的值 - 基于分辨率
    xp=(fwd_range[1]-fwd_range[0])/res
    yp = (side_range[1] - side_range[0]) / res

    # x_img = (x_points / xp).astype(np.int32)  # x axis is -y in LIDAR
    x_img = (-x_points / xp).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (y_points / yp).astype(np.int32)  # y axis is -x in LIDAR
    # y_img = (-y_points / yp).astype(np.int32)  # y axis is -x in LIDAR


    #x_img += abs(int(res/fwd_range[0]))
    #y_img += abs(int(res/side_range[1]))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])
    # pixel_values = np.transpose(pixel_values)
    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + res
    y_max = 1 + res
    im = np.zeros([x_max, y_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    # im[y_img, x_img] = pixel_values
    im[x_img, y_img] = pixel_values
    save = im[0:x_max-1, 0:y_max-1]
    # save=np.transpose(save)
    return save




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

def scale_to_255(a, min, max, dtype=np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def singlechannelfeature(PointCloud_, BoundaryCond, Discretization):
    Height = Discretization   #
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
    # !!!!!some important problem is image coordinate is (y,x), not (x,y)调整方向调整这个
    y = np.int_(PointCloud_remain[:, 0])    # x axis is -y in LIDAR
    # y = Discretization - y
    x = np.int_(PointCloud_remain[:, 1])    # y axis is -x in LIDAR
    x = Discretization -x

    # heightMap[y, x] = PointCloud_remain[:, 2]  # 高度作为数值
    heightMap[x,y] = PointCloud_remain[:, 2]

    #PointCloud_top = PointCloud[indices]
    # intensityMap[y, x] = PointCloud_remain[:, 3]  # 反射率
    intensityMap[x,y] = PointCloud_remain[:, 3]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    # densityMap[y, x] = normalizedCounts  # 用的是出现的次数作为对应坐标的数值
    densityMap[x,y] = normalizedCounts

    # 对三个通道的值进行归一化,后期输出的话可能要×255
    densityMap = densityMap / densityMap.max()
    heightMap = heightMap / heightMap.max()
    intensityMap = intensityMap / intensityMap.max()
    # densityMap = 255*densityMap / densityMap.max()
    # heightMap = 255*heightMap / heightMap.max()
    # intensityMap = 255*intensityMap / intensityMap.max()

    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap  # r_map
    RGB_Map[:, :, 1] = heightMap  # g_map
    RGB_Map[:, :, 2] = intensityMap  # b_map

    save = np.zeros((Discretization, Discretization, 3))
    save = RGB_Map[0:Discretization, 0:Discretization, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return save

def singleframe():
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.9, 'maxZ': 2.6, }
    pixel=480
    dataset=1 #2

    if dataset==1:
        father_path = 'F:/daxie/daxie/data/first'
        velosingle=True # first=True second=False
        s=[1,31]
    elif dataset==2:
        father_path = 'F:/daxie/daxie/data/second'
        velosingle = False  # first=True second=False
        s = [0, 33]

    for seq in range(s[0],s[1]):
        save_path_3c = os.path.join(father_path,'birdview/feature_3c/%d' %seq)
        save_path_1c = os.path.join(father_path,'birdview/feature_1c/%d'%seq)
        save_path_4c = os.path.join(father_path,'birdview/feature_4c/%d'%seq) #plt自动添加透明图层

        if not os.path.exists(save_path_3c):
            os.makedirs(save_path_3c)
        if not os.path.exists(save_path_1c):
            os.makedirs(save_path_1c)
        if not os.path.exists(save_path_4c):
            os.makedirs(save_path_4c)

        if velosingle:
            data_path = os.path.join(father_path,'velo/%d' %seq)
        else:
            data_path = os.path.join(father_path,'velo/%d/VLP160' %seq)

        list=os.listdir(data_path)
        for frame in list:
            print(frame)
            if velosingle:
            # 提取点云
                velo = load_velo_scan(os.path.join(data_path,frame)) #combine_pointCloud.load_2velo_sacn(father_path,frame_id)
            else:
                velo_0_path =os.path.join(data_path, frame)
                velo_0=feature_projection.load_velo_scan_raw(velo_0_path)
                velo_1 = feature_projection.load_velo_scan_raw(velo_0_path.replace("VLP160","VLP161"))
                velo = np.vstack((velo_0, velo_1))[:, :4]


            # 将点云转换为鸟瞰图并保存
            feature_map_1c = point_cloud_2_birdseye(velo, res=pixel, side_range=(-30, 30),
                                                             fwd_range=(-10, 50), height_range=(-1.9, 2.6))

            cv2.imwrite(save_path_1c + '/%d.png' % int(frame.split('.')[0]), feature_map_1c)
            #plt.imsave(save_path_1c + '/_%d.png' % int(frame.split('.')[0]), feature_map_1c,cmap=cm.gray)
            # 消除多余点
            velo = removePoints(velo, boundary)

            # 平移velo 将boundary的最左下角作为坐标原点，防止转换为feature map时出现负值
            velo[:, 0] += abs(boundary['minX'])
            velo[:, 1] += abs(boundary['minY'])
            velo[:, 2] += abs(boundary['minZ'])

            # 转转换为平面图像
            feature_map_3c = makeBVFeature(velo, boundary, pixel)
            plt.imsave(save_path_4c + '/%d.png' % int(frame.split('.')[0]), feature_map_3c)
            scale_255=np.ones((pixel,pixel,3))*255
            feature_map_3c=np.multiply(scale_255,feature_map_3c)
            cv2.imwrite(save_path_3c + '/%d.png' % int(frame.split('.')[0]), feature_map_3c)

def get_imu_data():
    """
    获取imu的数据
    :return: 以第一帧位置为原点的x坐标 y坐标 航向角
    """
    origin = None
    # 获取第一个filepath
    file_path = yield
    # 开始之后每次返回对应的新imu数据
    while file_path is not None:
        f_csv = csv.reader(open(file_path, 'r'), delimiter=' ')
        lat, lon, direction, v, rtk = next(f_csv)
        p1 = pyproj.Proj(init="epsg:4610")  # 定义数据地理坐标系
        p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
        x1, y1 = p1(float(lon), float(lat))
        x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
        # 计算相对于起点的偏移
        if origin is None:
            origin = (x2, y2)
        xy_scale_factor = 0.8505  # 修正坐标转换的误差
        # xy_scale_factor = 1  # 修正坐标转换的误差
        x, y = (x2 - origin[0]) * xy_scale_factor, (y2 - origin[1]) * xy_scale_factor
        file_path = yield x, y, float(direction)
'''
def get_window_merged(start_pos, length=3):
    """
    获取重叠几帧的数据
    :param start_pos:
    :param length:
    :return:
    """
    merged = []
    steps = []
    # 读取一系列帧数据
    for i in range(start_pos, start_pos + length):
        # 获取点云数据
        velo = load_velo_scan(os.path.join(velo_path, '%d.bin' % i))[:, :3]
        # 获取imu数据
        abs_x, abs_y, heading = imu_data_processer.send(os.path.join(imu_path, '%d.txt' % i))
        # 旋转并平移
        if length > 1:
            velo = rotate(velo, heading)
            velo[:, 0] = abs_x + velo[:, 0]
            velo[:, 1] = abs_y + velo[:, 1]
        merged.append(velo[:, :3])
        steps.append(np.ones([velo.shape[0], 1]) * (i * 128))
    # 合并多帧
    merged = np.vstack(merged)
    steps = np.vstack(steps).reshape(-1)
    return merged, steps
'''

if __name__ == '__main__':
    singleframe()

    #multiframe
    '''
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.6, 'maxZ': 2.4, }
    path = '/media/kid/workspace/daxie/output/birdview'

    for seq_id in range(11, 12):  # 1-30
        # seq_id = 2
        # 初始化imu数据

        father_path = '/media/kid/workspace/daxie/output/birdview/'
        save_path_3c = './output/feature_3c/%d' % seq_id
        save_path_1c = './output/feature_1c/%d' % seq_id

        if not os.path.exists(save_path_3c):
            os.mkdir(save_path_3c)
        if not os.path.exists(save_path_1c):
            os.mkdir(save_path_1c)

        #imu_path = os.path.join(path, 'imuseq/%d' % seq_id)
        # imu数据处理器
        #imu_data_processer = get_imu_data()
        #next(imu_data_processer)

        velo_path = os.path.join(path, 'veloseq/%d' % seq_id)
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
            feature_map_1c = top_view.point_cloud_2_birdseye(velo, res=512, side_range=(-30, 30),
                                                             fwd_range=(-10, 50), height_range=(-1.6, 2.4))
            plt.imsave(save_path_1c + '/%d.png' % frame_id, feature_map_1c)

            velo = removePoints(velo, boundary)

            # 平移velo 将boundary的最左下角作为坐标原点，防止转换为feature map时出现负值
            velo[:, 0] += abs(boundary['minX'])
            velo[:, 1] += abs(boundary['minY'])
            velo[:, 2] += abs(boundary['minZ'])

            # 转转换为平面图像
            feature_map_3c = makeBVFeature(velo, boundary, 512)
            plt.imsave(save_path_3c + '/%d.png' % frame_id, feature_map_3c)
    '''

