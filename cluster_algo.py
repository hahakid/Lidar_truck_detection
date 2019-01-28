import functools
import os
import csv

import mayavi
import pyproj
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from PointcloudVoxelizer.source import pointclouds_to_voxelgrid
from funtiontest import load_velo_scan, cmp
from mayavi import mlab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def filter_ground(point_cloud_mat, grid_size=3):
    """
    过滤掉大地
    :param point_cloud_mat: 点云
    :param grid_size: 网格大小
    :return:
    """
    for i in range(0, point_cloud_mat.shape[0], grid_size):
        for j in range(0, point_cloud_mat.shape[1], grid_size):
            z = np.where(point_cloud_mat[i:i + grid_size, j:j + grid_size, :] == 1)[2]
            if len(z) > 0:
                # if max(z) - min(z) < point_cloud_mat.shape[2] / 3:
                #     point_cloud_mat[i:i + grid_size, j:j + grid_size, :] = 0
                #     print(i, j, z)
                if max(z) < point_cloud_mat.shape[2] / 1.8:
                    point_cloud_mat[i:i + grid_size, j:j + grid_size, :] = 0
                # else:
                #     point_cloud_mat[i:i + grid_size, j:j + grid_size, :] = 1
    return point_cloud_mat


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
        # xy_scale_factor = 1  # 修正坐标转换的误差
        x, y = (x2 - origin[0]) * xy_scale_factor, (y2 - origin[1]) * xy_scale_factor
        file_path = yield x, y, float(direction)


def rotate(point_cloud, angle):
    """
    以z轴为中心旋转angle角度
    :param point_cloud:
    :param angle:
    :return:
    """
    theta = angle / 180 * np.pi
    R_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    point_cloud_rotated = np.matmul(R_matrix, point_cloud.T).T
    return point_cloud_rotated


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
        # # 自身的雷达波
        # ego_range = 15
        # velo = velo[np.where(
        #     (velo[:, 0] > ego_range) |
        #     (velo[:, 0] < -ego_range) |
        #     (velo[:, 1] > ego_range) |
        #     (velo[:, 1] < -ego_range)
        # )]
        # 获取imu数据
        abs_x, abs_y, heading = imu_data_processer.send(os.path.join(imu_path, '%d.txt' % i))
        # 旋转并平移
        velo = rotate(velo, heading)
        velo[:, 0] = abs_x + velo[:, 0]
        velo[:, 1] = abs_y + velo[:, 1]
        merged.append(velo[:, :3])
        steps.append(np.ones([velo.shape[0], 1]) * (i * 128))
    # 合并多帧
    merged = np.vstack(merged)
    steps = np.vstack(steps).reshape(-1)
    return merged, steps


def overlap_voxelization(point_cloud, overlap_frame_count, voxel_granularity=400):
    """
    计算两个重叠密度增大之后的帧之间的voxel差异
    :param point_cloud: 第一帧的位置
    :param frame2: 第二帧的位置
    :param overlap_frame_count: 重叠帧数
    :param voxel_granularity: voxel的粒度
    :return:
    """
    # 获得融合后的第一帧
    pc_frame1, _ = get_window_merged(start_pos=point_cloud, length=overlap_frame_count)
    # 计算体素
    voxel_converter = pointclouds_to_voxelgrid.data_loader(point_list=pc_frame1)
    _, _, voxel, _, _ = voxel_converter(xyz_range=BORDER_TH, mag_coeff=voxel_granularity)
    # 过滤地面
    voxel = filter_ground(voxel)

    return voxel


def test_slam():
    pc_frame1, steps = get_window_merged(start_pos=180, length=100)
    mayavi.mlab.points3d(pc_frame1[:, 0], pc_frame1[:, 1], pc_frame1[:, 2],
                         # mayavi.mlab.points3d(xx, yy, zz,
                         -pc_frame1[:, 2],  # Values used for Color
                         # mode="cube",
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         figure=fig,
                         # color=(1, 0, 0),  # Used a fixed (r,g,b) instead
                         scale_factor=1,
                         )
    mayavi.mlab.show()
    exit()


def process_clustering_result(xx, yy, zz, cl_rst, fig):
    # 删除标签为-1的噪声点
    pc = np.vstack([xx, yy, zz, cl_rst]).T
    # pc = pc[np.where(cl_rst != -1)]
    represents_list = []
    for category_id in set(cl_rst) - {-1}:
        # 查找当前类别的
        points = pc[np.where(cl_rst == category_id)]
        # 计算边框最值
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        y_min = points[:, 1].min()
        y_max = points[:, 1].max()
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        # 删除底面积过小的块
        if (abs(x_max - x_min) / voxel_scale) * (abs(y_max - y_min) / voxel_scale) < 2: continue
        # 绘制bounding box
        # bounding_box = np.vstack([
        #     # 底部平面
        #     [x_min, y_min, z_min],
        #     [x_min, y_max, z_min],
        #     [x_max, y_max, z_min],
        #     [x_max, y_min, z_min],
        #     [x_min, y_min, z_min],
        #     # 上部平面
        #     [x_min, y_min, z_max],
        #     [x_min, y_max, z_max],
        #     [x_max, y_max, z_max],
        #     [x_max, y_min, z_max],
        #     [x_min, y_min, z_max],
        # ])
        # 　顶部底部平面
        # mlab.plot3d(
        #     bounding_box[0:10, 0],
        #     bounding_box[0:10, 1],
        #     bounding_box[0:10, 2],
        #     line_width=20,
        #     tube_radius=0.1,
        #     tube_sides=12
        # )
        # # 侧面三条线
        # for i in range(1, 4):
        #     mlab.plot3d(
        #         bounding_box[[i, i + 5], 0],
        #         bounding_box[[i, i + 5], 1],
        #         bounding_box[[i, i + 5], 2],
        #         line_width=20,
        #         tube_radius=0.1,
        #         tube_sides=12
        #     )
        # represents_list.append([
        #     (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2,  # XYZ坐标
        #     category_id,  # 类别ID
        #     0, 0  # 新点的Vx和Vy速度均为0
        # ])
        # 使用算术中心
        represents_list.append([
            np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2]),  # XYZ坐标
            category_id + frame_id,  # 类别ID+帧ID 得到唯一ID
            0, 0,  # 新点的Vx和Vy速度均为0,
            0,  # 生命周期
            0,  # 累计速度
        ])
    represents_list = np.array(represents_list)
    return represents_list


def track(prev, cur, distance_th, beta1, beta2):
    """
    跟踪目标
    目标列表矩阵的结构: [x,y,z,类别,Vx,Vy,Life,TotalDis]
    :param prev: 前一帧的目标列表
    :param cur: 当前帧检测算出来的中心
    :param distance_th: 两帧之间最近点的最远距离阈值
    :param beta1: 位置估计权重衰减值
    :param beta2: 速度估计权重衰减值
    :return: 当前帧的目标列表
    """
    center_list = []
    for prev_center in prev:
        print('PREV: ', prev_center)
        min_distance = 99999999
        min_distance_center_idx = None
        # 查找最短距离点
        for idx, cur_center in enumerate(cur):
            # 如果当前这个点没有被别的点选中
            if cur_center[3] != np.inf:
                # 比较最短距离
                distance = np.sqrt(
                    (prev_center[0] - cur_center[0]) ** 2 +
                    (prev_center[1] - cur_center[1]) ** 2 +
                    (prev_center[2] - cur_center[2]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    min_distance_center_idx = idx
        # 据当前点最短点的距离小于阈值 即目标没消失
        if min_distance < distance_th:
            print('MIN: ', cur[min_distance_center_idx])

            # 对xy坐标进行指数加权平均 beta * (prev + V * t) + (1 - beta) * cur
            cur[min_distance_center_idx][:2] = \
                beta1 * (prev_center[0:2] + FRAME_TIME_INTERVAL * prev_center[4:6]) \
                + (1 - beta1) * (cur[min_distance_center_idx][:2])

            # 当前最近点的label设置为与上一个点相同 仅坐标为当前点的
            cur[min_distance_center_idx][3] = prev_center[3]

            # 计算速度
            # Vx
            Vx = (cur[min_distance_center_idx][0] - prev_center[0]) / voxel_scale / FRAME_TIME_INTERVAL
            cur[min_distance_center_idx][4] = beta2 * prev_center[4] + (1 - beta2) * Vx
            # Vy
            Vy = (cur[min_distance_center_idx][1] - prev_center[1]) / voxel_scale / FRAME_TIME_INTERVAL
            cur[min_distance_center_idx][5] = beta2 * prev_center[5] + (1 - beta2) * Vy

            # 生命周期增加
            cur[min_distance_center_idx][6] = prev_center[6] + 1

            # 累计速路程增加
            cur[min_distance_center_idx][7] = \
                prev_center[7] + np.sqrt(
                    cur[min_distance_center_idx][4] ** 2 + cur[min_distance_center_idx][5] ** 2
                ) * FRAME_TIME_INTERVAL

            # 加入当前帧的结果列表
            center_list.append(cur[min_distance_center_idx].tolist())
            # 标记当前位置已经选择过
            cur[min_distance_center_idx][3] = np.inf
        else:
            # 如果上一帧的目标消失了 仅当上一帧生命周期消失了再删除 否则仍加入下一帧的tracking list里
            if prev_center[6] > 0:
                # 生命周期减少
                prev_center[6] -= 1
                # 用速度预估下一帧位置
                prev_center[0:2] += FRAME_TIME_INTERVAL * prev_center[4:6]
                prev_center[7] += np.sqrt(prev_center[4] ** 2 + prev_center[5] ** 2) * FRAME_TIME_INTERVAL
                center_list.append(prev_center.tolist())
    # 把当前列表中的 从前没有出现过的新节点添加到当前帧
    # TODO 新增的中心点要有新的标号
    for cur_center in cur:
        if cur_center[3] != np.inf:
            print('NEW: ', cur_center)
            center_list.append(list(cur_center))
    print('=' * 20)
    # return np.vstack(center_list)
    return np.array(center_list)


if __name__ == '__main__':
    # 帧时间间隔(s)
    FRAME_TIME_INTERVAL = 0.1

    # 重叠的帧数
    OVERLAP_FRAME_COUNT = 2

    # 体素边界
    BORDER_TH = (-60, -50, -1.9, 60, 50, 2.4)

    # 体素化粒度
    VOXEL_GRANULARITY = 400

    # 真实世界尺度(米)对应的voxel格子数
    voxel_scale = VOXEL_GRANULARITY / (BORDER_TH[3] - BORDER_TH[0])

    # 聚类参数
    CLUSTERING_EPS = 1.3
    CLUSTERING_MIN_SP = 5

    # 近邻传播的距离阈值
    NN_DISTANCE_TH = 2.5

    # 指数加权平均滤波的阈值 pos为位置 v为速度
    BETA_POS = 0.3
    BETA_V = 0.75

    # 追踪并可视化
    for seq_id in range(1, 31):
        # 初始化imu数据
        imu_path = './data/imuseq/%d' % seq_id
        # imu数据处理器
        imu_data_processer = get_imu_data()
        next(imu_data_processer)
        # 获得多帧融合之后的点云
        velo_path = './data/veloseq/%d' % seq_id

        # 上一帧的结果
        last_frame_tracking_list = []
        # 计算多帧的聚类结果
        fig = mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640 * 2, 360 * 2))
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cmp)):
            print(frame_file)
            frame_id = int(frame_file.replace('.bin', ''))
            if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + OVERLAP_FRAME_COUNT))):
                break
            # 重叠并体素化
            overlapped_pc_mat = overlap_voxelization(
                point_cloud=frame_id,
                overlap_frame_count=OVERLAP_FRAME_COUNT,
                voxel_granularity=VOXEL_GRANULARITY
            )
            xx, yy, zz = np.where(overlapped_pc_mat > 0)

            # 放缩z轴
            zz = np.array(zz, dtype=np.float32) * 0.01

            if len(xx) == 0: continue

            # 聚类
            clustering = DBSCAN(
                eps=CLUSTERING_EPS * voxel_scale,
                min_samples=CLUSTERING_MIN_SP,
                n_jobs=-1).fit(
                np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
            )
            cl_rst = np.array(clustering.labels_)

            # 处理聚类结果
            tracking_list = process_clustering_result(xx, yy, zz, cl_rst, fig)

            # 与上一帧的结果对比并追踪目标
            if len(last_frame_tracking_list) > 0:
                tracking_list = track(
                    prev=last_frame_tracking_list, cur=tracking_list,
                    distance_th=NN_DISTANCE_TH * voxel_scale,
                    beta1=BETA_POS,
                    beta2=BETA_V
                )
                print('LABELS: ', tracking_list[:, 3])
                print('=' * 20)

            last_frame_tracking_list = tracking_list

            # 可视化结果
            # 仅可视化出现3帧以上的目标
            tracking_list = tracking_list[
                np.where(
                    tracking_list[:, 6] > 2
                )
            ]
            if len(tracking_list) == 0: continue
            nodes = mayavi.mlab.points3d(
                tracking_list[:, 0],
                tracking_list[:, 1],
                tracking_list[:, 2] * 100,
                # clustering.labels_,
                # mode="cube",
                mode="cube",
                # color=(0, 1, 0),
                # vmax=100,
                colormap='spectral',
                figure=fig,
                scale_factor=3
            )
            for center in tracking_list:
                mayavi.mlab.text3d(
                    center[0] + 2,
                    center[1] + 2,
                    center[2] * 100 + 2,
                    '%.0f (%.1f, %.1f)' % (
                        center[3],
                        np.sqrt(center[4] ** 2 + center[5] ** 2),
                        np.arctan(center[4] / center[5]),
                    ),
                    scale=5,
                    figure=fig,
                )
            nodes.glyph.scale_mode = 'scale_by_vector'
            nodes.mlab_source.dataset.point_data.scalars = tracking_list[:, 3] / max(tracking_list[:, 3])
            # nodes.mlab_source.dataset.point_data.scalars = centers[:, 7] / max(centers[:, 7])

            # 可视化原始点云
            xx, yy, zz = np.where(overlapped_pc_mat > 0)
            mayavi.mlab.points3d(
                xx, yy, zz,
                # clustering.labels_,
                # mode="cube",
                mode="point",
                # color=(0, 1, 0),
                colormap='spectral',
                figure=fig,
                scale_factor=1
            )
            # mlab.outline()
            # xy平面0度 z轴0度 摄像机离xy平面300米 焦点在整个图形中点
            mlab.view(0, 30, 640, focalpoint=(overlapped_pc_mat.shape[0] / 2, overlapped_pc_mat.shape[1] / 2, 0))
            # 创建文件夹并保存
            if not os.path.exists('./output/%d' % (seq_id)):
                os.mkdir('./output/%d' % (seq_id))
            mlab.savefig(filename='./output/%d/%d.png' % (seq_id, frame_id), figure=fig)
            mlab.clf()
