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
                if max(z) < point_cloud_mat.shape[2] / 2:
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


def calc_frame_diff(frame1, frame2, overlap_frame_count, voxel_granularity=400):
    """
    计算两个重叠密度增大之后的帧之间的voxel差异
    :param frame1: 第一帧的位置
    :param frame2: 第二帧的位置
    :param overlap_frame_count: 重叠帧数
    :param voxel_granularity: voxel的粒度
    :return:
    """
    # 获得融合后的第一帧
    pc_frame1, _ = get_window_merged(start_pos=frame1, length=overlap_frame_count)
    # 计算体素
    voxel_converter = pointclouds_to_voxelgrid.data_loader(point_list=pc_frame1)
    _, _, voxel1, _, _ = voxel_converter(xyz_range=border_thresh, mag_coeff=voxel_granularity)
    # 过滤地面
    voxel1 = filter_ground(voxel1)

    # 获得融合后的第二帧
    pc_frame2, _n = get_window_merged(start_pos=frame2, length=overlap_frame_count)
    # 计算体素
    voxel_converter = pointclouds_to_voxelgrid.data_loader(point_list=pc_frame2)
    _, _, voxel2, _, _ = voxel_converter(xyz_range=border_thresh, mag_coeff=voxel_granularity)
    # 过滤地面
    voxel2 = filter_ground(voxel2)
    # 对比两个点云
    compared_mat = np.logical_and(np.logical_not(voxel1), voxel2)

    # xx, yy, zz = np.where(voxel1 > 0)
    # mayavi.mlab.points3d(xx, yy, zz,
    #                      # clustering.labels_,
    #                      # mayavi.mlab.points3d(xx, yy, zz,
    #                      # steps,  # Values used for Color
    #                      mode="point",
    #                      colormap='spectral',  # 'bone', 'copper', 'gnuplot'
    #                      figure=fig,
    #                      color=(1, 1, 1),  # Used a fixed (r,g,b) instead
    #                      scale_factor=1,
    #                      )

    return compared_mat, voxel1, voxel2


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
        # 删除体积过小的块
        if abs(x_max - x_min) * abs(y_max - y_min) * abs(z_max - z_min) < 2: continue
        # 绘制bounding box
        bounding_box = np.vstack([
            # 底部平面
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_max, z_min],
            [x_max, y_min, z_min],
            [x_min, y_min, z_min],
            # 上部平面
            [x_min, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_max, z_max],
            [x_max, y_min, z_max],
            [x_min, y_min, z_max],
        ])
        # 　顶部底部平面
        mlab.plot3d(
            bounding_box[0:10, 0],
            bounding_box[0:10, 1],
            bounding_box[0:10, 2],
            line_width=20,
            tube_radius=0.1,
            tube_sides=12
        )
        # 侧面三条线
        for i in range(1, 4):
            mlab.plot3d(
                bounding_box[[i, i + 5], 0],
                bounding_box[[i, i + 5], 1],
                bounding_box[[i, i + 5], 2],
                line_width=20,
                tube_radius=0.1,
                tube_sides=12
            )
        represents_list.append([
            (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2, category_id
        ])
    represents_list = np.array(represents_list)
    # clustering = DBSCAN(eps=2.5, min_samples=1, n_jobs=-1).fit(represents_list[:, 0:3])
    # represents_list[:, 3] = np.array(clustering.labels_)
    return represents_list


def draw_origin(frame, fig):
    xx, yy, zz = np.where(frame > 0)
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


if __name__ == '__main__':
    # 可视化
    # 体素边界
    border_thresh = (-50, -60, -1.9, 50, 60, 2.4)
    seq_id = 1
    # 初始化imu数据
    imu_path = './data/imuseq/%d' % seq_id
    # imu数据处理器
    imu_data_processer = get_imu_data()
    next(imu_data_processer)
    # 获得多帧融合之后的点云
    velo_path = './data/veloseq/%d' % seq_id

    # 计算多帧的聚类结果
    for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cmp)):
        fig = mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640 * 2, 360 * 2))
        print(frame_file)
        frame_id = int(frame_file.replace('.bin', ''))
        if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + 3 + 3))):
            break
        if frame_id < 230:
            continue
        # 计算两帧diff
        compared_mat, frame1, _ = calc_frame_diff(frame1=frame_id, frame2=frame_id + 3, overlap_frame_count=2)
        # xx, yy, zz = np.where(compared_mat > 0)
        xx, yy, zz = np.where(frame1 > 0)
        # 放缩z轴
        zz = np.array(zz, dtype=np.float32) * 0.1

        # 聚类
        clustering = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1).fit(
            np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
        )
        cl_rst = np.array(clustering.labels_)
        # 处理聚类结果
        centers = process_clustering_result(xx, yy, zz * 10, cl_rst, fig)
        nodes = mayavi.mlab.points3d(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            # clustering.labels_,
            # mode="cube",
            mode="cube",
            # color=(0, 1, 0),
            colormap='spectral',
            figure=fig,
            scale_factor=3
        )
        nodes.glyph.scale_mode = 'scale_by_vector'
        nodes.mlab_source.dataset.point_data.scalars = centers[:, 3] / max(centers[:, 3])
        xx, yy, zz = np.where(frame1 > 0)
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
        # mlab.show()
        # xy平面0度 z轴0度 摄像机离xy平面300米 焦点在整个图形中点
        mlab.view(0, 30, 540, focalpoint=(compared_mat.shape[0] / 2, compared_mat.shape[1] / 2, 0))
        # 创建文件夹并保存
        if not os.path.exists('./output/%d' % (seq_id)):
            os.mkdir('./output/%d' % (seq_id))
        mlab.savefig(filename='./output/%d/%d.png' % (seq_id, frame_id), figure=fig)
        mlab.clf()
