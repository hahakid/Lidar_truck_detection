import functools
import os
import csv
import time

import mayavi
import pyproj
import numpy as np
from sklearn.cluster import DBSCAN

from functions.utils import load_velo_scan, cmp
from mayavi import mlab
import pylab as plt
import cv2
import pcl


class ClusterDetector():
    # 全局参数
    # 帧时间间隔(s)
    FRAME_TIME_INTERVAL = 0.1

    # 重叠的帧数
    OVERLAP_FRAME_COUNT = 2

    # 点云边界 x1,y1,1 x2,y2,z2
    BORDER_TH = (-40, -50, -1.9, 40, 50, 2.4)
    GROUND_LIMIT = (0.8, 5)

    # 体素化粒度
    VOXEL_GRANULARITY = 400
    # voxel下采样滤波的栅格尺寸
    VOXEL_GRID_SIZE = 0.3

    # 是否过滤地面
    IS_FILTER_GROUND = True
    # 过滤地面所使用的网格算法的网格大小
    FG_GRID_SIZE = 3

    # 聚类参数
    CLUSTERING_EPS = 1.3
    CLUSTERING_MIN_SP = 5

    # 某个类别的最小投影面积
    MIN_XY_PLATE = 2
    # 最长的车身长度 大于这个长度的就被过滤
    MAX_TRUCK_LENGTH = 20

    # 近邻传播的距离阈值
    NN_DISTANCE_TH = 2.5

    # 指数加权平均滤波的阈值 pos为位置 v为速度
    BETA_POS = 0.3
    BETA_V = 0.75

    # 是否可视化
    VISUALIZE = True

    # 存储接收到的点云队列
    PC_QUEUE = []

    # 真实世界尺度(米)对应的voxel格子数
    # voxel_scale = VOXEL_GRANULARITY / (BORDER_TH[3] - BORDER_TH[0])
    voxel_scale = 1

    def __init__(self):
        # 开始行驶时在大地坐标系下车体原点坐标
        self.ORIGIN = None
        self.cur_overlapped_pc_mat = np.array([])
        # 初始化绘图的figure
        self.fig = mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640 * 2, 360 * 2)) \
            if ClusterDetector.VISUALIZE else None

    def filter_ground(self, point_cloud_mat, grid_size=3):
        """
        过滤掉地面
        :param point_cloud_mat: 点云
        :param grid_size: 网格大小
        :return: 过滤掉地面后的点云
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

    # def get_imu_data(self):
    #     """
    #     获取imu的数据
    #     :return: 以第一帧位置为原点的x坐标 y坐标 航向角
    #     """
    #     origin = None
    #     # 获取第一个filepath
    #     file_path = yield
    #     # 开始之后每次返回对应的新imu数据
    #     while file_path is not None:
    #         f_csv = csv.reader(open(file_path, 'r'), delimiter=' ')
    #         lat, lon, direction, v, rtk = next(f_csv)
    #         p1 = pyproj.Proj(init="epsg:4610")  # 定义数据地理坐标系
    #         p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
    #         x1, y1 = p1(float(lon), float(lat))
    #         x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    #         # 计算相对于起点的偏移
    #         if origin is None:
    #             origin = (x2, y2)
    #         xy_scale_factor = 0.8505  # 修正坐标转换的误差
    #         # xy_scale_factor = 1  # 修正坐标转换的误差
    #         # xy_scale_factor = 1  # 修正坐标转换的误差
    #         x, y = (x2 - origin[0]) * xy_scale_factor, (y2 - origin[1]) * xy_scale_factor
    #         file_path = yield x, y, float(direction)

    def process_imu_data(self, lat, lon, direction, v, rtk):
        """
        将imu数据转换为大地坐标
        :param lat: 精度
        :param lon: 维度
        :param direction: 航向角
        :param v: 速度
        :param rtk: RTK状态
        :return: (x,y,航向)
        """
        p1 = pyproj.Proj(init="epsg:4610")  # 定义数据地理坐标系
        p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
        x1, y1 = p1(float(lon), float(lat))
        x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
        # 计算相对于起点的偏移
        if self.ORIGIN is None:
            self.ORIGIN = (x2, y2)
        xy_scale_factor = 0.8505  # 修正坐标转换的误差
        # xy_scale_factor = 1  # 修正坐标转换的误差
        # xy_scale_factor = 1  # 修正坐标转换的误差
        x, y = (x2 - self.ORIGIN[0]) * xy_scale_factor, (y2 - self.ORIGIN[1]) * xy_scale_factor
        return x, y, float(direction)

    def rotate(self, point_cloud, angle):
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

    def get_window_merged(self, start_pos, length=3):
        """
        获取重叠几帧的数据
        :param start_pos:
        :param length:
        :return:
        """
        merged = []
        imu_data = []
        # 读取一系列帧数据
        for i in range(start_pos, start_pos + length):
            # 获取点云数据
            velo, imu = ClusterDetector.PC_QUEUE[i]
            # 处理并转换imu数据
            abs_x, abs_y, heading = self.process_imu_data(*imu)
            # 旋转XYZ坐标 保留反射率
            velo = np.hstack([self.rotate(velo[:, :3], heading), velo[:, :-1]])
            # 平移
            velo[:, 0] = abs_x + velo[:, 0]
            velo[:, 1] = abs_y + velo[:, 1]
            merged.append(velo[:, :4])

            # 存储第一帧的imu数据
            if len(imu_data) == 0:
                imu_data = np.array([abs_x, abs_y, heading])
        # 合并多帧
        merged = np.vstack(merged)
        return merged, imu_data

    def overlap_voxelization(self, point_cloud_id, overlap_frame_count, voxel_granularity=400):
        """
        计算两个重叠密度增大之后的帧之间的voxel差异
        :param point_cloud_id: 第一帧的位置
        :param frame2: 第二帧的位置
        :param overlap_frame_count: 重叠帧数
        :param voxel_granularity: voxel的粒度
        :return: 体素化之后的叠加点云, 当前帧的imu数据
        """
        # 获得融合后的第一帧
        merged_frame, imu_mat = self.get_window_merged(start_pos=point_cloud_id, length=overlap_frame_count)
        # 转化为pcl点云
        cloud = pcl.PointCloud_PointXYZI(np.asarray(merged_frame, dtype=np.float32))
        if ClusterDetector.IS_FILTER_GROUND:
            # 过滤掉区域外部的点云 以及地面
            passthrough = cloud.make_passthrough_filter()
            passthrough.set_filter_field_name("x")
            passthrough.set_filter_limits(ClusterDetector.BORDER_TH[0], ClusterDetector.BORDER_TH[3])
            passthrough.set_filter_field_name("y")
            passthrough.set_filter_limits(ClusterDetector.BORDER_TH[1], ClusterDetector.BORDER_TH[4])
            passthrough.set_filter_field_name("z")
            passthrough.set_filter_limits(ClusterDetector.GROUND_LIMIT[0], ClusterDetector.GROUND_LIMIT[1])
            cloud = passthrough.filter()
        # 进行voxel滤波
        sor = cloud.make_voxel_grid_filter()
        sor.set_leaf_size(ClusterDetector.VOXEL_GRID_SIZE, ClusterDetector.VOXEL_GRID_SIZE,
                          ClusterDetector.VOXEL_GRID_SIZE)
        voxel = np.asarray(sor.filter())
        # 过滤地面
        # voxel = filter_ground(voxel, grid_size=FG_GRID_SIZE)

        return voxel, imu_mat

    def test_slam(self):
        pc_frame1, steps = self.get_window_merged(start_pos=180, length=100)
        mayavi.mlab.points3d(pc_frame1[:, 0], pc_frame1[:, 1], pc_frame1[:, 2],
                             # mayavi.mlab.points3d(xx, yy, zz,
                             -pc_frame1[:, 2],  # Values used for Color
                             # mode="cube",
                             mode="point",
                             colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                             figure=self.fig,
                             # color=(1, 0, 0),  # Used a fixed (r,g,b) instead
                             scale_factor=1,
                             )
        mayavi.mlab.show()
        exit()

    def process_clustering_result(self, xx, yy, zz, cl_rst, unique_id, fig):
        """
        处理过滤聚类的结果
        :param xx: x列表
        :param yy: y列表
        :param zz: z列表
        :param cl_rst: 聚类结果的类别列表
        :param fig: 可视化boundingbox的figure
        :return:
        """
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
            if (abs(x_max - x_min) / self.voxel_scale) * (
                    abs(y_max - y_min) / ClusterDetector.voxel_scale) < ClusterDetector.MIN_XY_PLATE: continue

            # 删除大于车身长度过长的块
            if np.sqrt((x_max - x_min) ** 2 + (
                    y_max - y_min) ** 2) > ClusterDetector.MAX_TRUCK_LENGTH * ClusterDetector.voxel_scale: continue
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
            # 计算最小外接矩形
            area_rect = cv2.minAreaRect(np.asarray(points[:, :2], dtype=np.int32))
            # 求4个顶点box
            box = cv2.boxPoints(area_rect)
            # 使用算术中心
            represents_list.append([
                np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2]),  # XYZ坐标
                category_id + unique_id,  # 类别ID+帧ID 得到唯一ID
                0, 0,  # 新点的Vx和Vy速度均为0,
                0,  # 生命周期
                0,  # 累计速度
                *box.reshape(-1)  # 最小外接矩阵box的四角坐标
            ])
        represents_list = np.array(represents_list)
        return represents_list

    def track(self, prev, cur, distance_th, beta1, beta2):
        """
        跟踪目标
        目标列表矩阵的结构: [x,y,z,目标ID,Vx,Vy,Life,TotalDis]
        :param prev: 前一帧的目标列表
        :param cur: 当前帧检测算出来的中心
        :param distance_th: 两帧之间最近点的最远距离阈值
        :param beta1: 位置估计权重衰减值
        :param beta2: 速度估计权重衰减值
        :return: 当前帧的目标列表 大小是n x 16
        每一行是16个元素
        [x,y,z,目标ID,Vx,Vy,Life,TotalDis,[4个外接矩形的顶点坐标(x1,y1)~(x4,y4)]]

        """
        center_list = []
        for prev_center in prev:
            # print('PREV: ', prev_center)
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
                # print('MIN: ', cur[min_distance_center_idx])

                # 对xy坐标进行指数加权平均 beta * (prev + V * t) + (1 - beta) * cur
                cur[min_distance_center_idx][:2] = \
                    beta1 * (prev_center[0:2] + ClusterDetector.FRAME_TIME_INTERVAL * prev_center[4:6]) \
                    + (1 - beta1) * (cur[min_distance_center_idx][:2])

                # 当前最近点的label设置为与上一个点相同 仅坐标为当前点的
                cur[min_distance_center_idx][3] = prev_center[3]

                # 计算速度
                # Vx
                Vx = (cur[min_distance_center_idx][0] - prev_center[
                    0]) / ClusterDetector.voxel_scale / ClusterDetector.FRAME_TIME_INTERVAL
                cur[min_distance_center_idx][4] = beta2 * prev_center[4] + (1 - beta2) * Vx
                # Vy
                Vy = (cur[min_distance_center_idx][1] - prev_center[
                    1]) / ClusterDetector.voxel_scale / ClusterDetector.FRAME_TIME_INTERVAL
                cur[min_distance_center_idx][5] = beta2 * prev_center[5] + (1 - beta2) * Vy

                # 生命周期增加
                cur[min_distance_center_idx][6] = prev_center[6] + 1

                # 累计路程增加
                cur[min_distance_center_idx][7] = \
                    prev_center[7] + np.sqrt(
                        cur[min_distance_center_idx][4] ** 2 + cur[min_distance_center_idx][5] ** 2
                    ) * ClusterDetector.FRAME_TIME_INTERVAL

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
                    prev_center[0:2] += ClusterDetector.FRAME_TIME_INTERVAL * prev_center[4:6]
                    prev_center[7] += np.sqrt(
                        prev_center[4] ** 2 + prev_center[5] ** 2) * ClusterDetector.FRAME_TIME_INTERVAL
                    center_list.append(prev_center.tolist())
        # 把当前列表中的 从前没有出现过的新节点添加到当前帧
        for cur_center in cur:
            if cur_center[3] != np.inf:
                # print('NEW: ', cur_center)
                center_list.append(list(cur_center))
        return np.array(center_list)

    def visualize_result(self, tracking_list, frame_id, seq_id):
        """
        可视化在点云中正在跟踪的目标
        :param tracking_list:
        :return:
        """
        if len(tracking_list) == 0:
            return
        # 仅可视化出现3帧以上的目标
        tracking_list = tracking_list[
            np.where(
                tracking_list[:, 6] > 2
            )
        ]
        if len(tracking_list) == 0: return
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
            figure=self.fig,
            scale_factor=2
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
                scale=3,
                figure=self.fig,
            )
        nodes.glyph.scale_mode = 'scale_by_vector'
        nodes.mlab_source.dataset.point_data.scalars = tracking_list[:, 3] / max(tracking_list[:, 3])
        # nodes.mlab_source.dataset.point_data.scalars = centers[:, 7] / max(centers[:, 7])

        # 可视化原始点云
        # xx, yy, zz = np.where(overlapped_pc_mat > 0)
        xx, yy, zz = \
            self.cur_overlapped_pc_mat[:, 0], self.cur_overlapped_pc_mat[:, 1], self.cur_overlapped_pc_mat[:, 2]

        mayavi.mlab.points3d(
            xx, yy, zz,
            # clustering.labels_,
            # mode="cube",
            mode="point",
            # color=(0, 1, 0),
            colormap='spectral',
            figure=self.fig,
            scale_factor=1
        )
        # mlab.outline()
        # xy平面0度 z轴0度 摄像机离xy平面300米 焦点在整个图形中点
        mlab.view(0, 30, 300, focalpoint=(0, 0, 0))
        # 创建文件夹并保存
        if not os.path.exists('./output/%d' % (seq_id)):
            os.mkdir('./output/%d' % (seq_id))
        mlab.savefig(filename='./output/%d/%d.png' % (seq_id, frame_id), figure=self.fig)
        mlab.clf()

    def update(self, point_cloud, imu, last_frame_tracking_list, unique_id):
        """
        接收新的帧并更新目标追踪结果
        :param point_cloud: 当前帧的点云
        :param imu: 当前帧的imu数据
        :param last_frame_tracking_list: 上一帧的追踪结果
        :return:
        1.如果达到累计重叠的帧数 返回新的追踪列表 否则返回之前未更新的追踪列表
        2.目标列表n x [4个角坐标,速度,角度]
        3.叠加后的点云
        """
        # 将接收到的点云数据和imu数据存储到队列中
        ClusterDetector.PC_QUEUE.append(
            [point_cloud, imu]
        )

        # 当前帧的车体在大地坐标系下的绝对坐标
        x, y, heading = self.process_imu_data(*imu)
        cur_car_position = np.array([x, y])

        # 如果当前已接收的帧小于叠加的帧 直接返回之前的跟踪结果
        if len(ClusterDetector.PC_QUEUE) < ClusterDetector.OVERLAP_FRAME_COUNT:
            return last_frame_tracking_list, np.array([])

        # 使用接收到的点云数据进行目标检测以及追踪
        # 重叠并体素化 同时获取第一帧的imu数据 以在未来做逆变换
        cur_overlapped_pc_mat, imu_mat = self.overlap_voxelization(
            point_cloud_id=0,  # 从队首开始取数据
            overlap_frame_count=ClusterDetector.OVERLAP_FRAME_COUNT,
            voxel_granularity=ClusterDetector.VOXEL_GRANULARITY
        )
        xx, yy, zz = \
            cur_overlapped_pc_mat[:, 0], cur_overlapped_pc_mat[:, 1], cur_overlapped_pc_mat[:, 2]
        # 保存重叠后的点云 用来可视化
        self.cur_overlapped_pc_mat = cur_overlapped_pc_mat
        # 放缩z轴
        zz = np.array(zz, dtype=np.float32) * 0.01

        if len(xx) == 0: return last_frame_tracking_list

        # 聚类
        clustering = DBSCAN(
            eps=ClusterDetector.CLUSTERING_EPS * ClusterDetector.voxel_scale,
            min_samples=ClusterDetector.CLUSTERING_MIN_SP,
            n_jobs=1
        ).fit(
            np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
        )
        cl_rst = np.array(clustering.labels_)

        # 处理聚类结果
        tracking_list = self.process_clustering_result(xx, yy, zz, cl_rst, unique_id, self.fig)

        # 与上一帧的结果对比并追踪目标
        if len(last_frame_tracking_list) > 0:
            tracking_list = self.track(
                prev=last_frame_tracking_list, cur=tracking_list,
                distance_th=ClusterDetector.NN_DISTANCE_TH * ClusterDetector.voxel_scale,
                beta1=ClusterDetector.BETA_POS,
                beta2=ClusterDetector.BETA_V
            )

        # tracking list的坐标均为大地坐标系
        # print('=' * 15, 'FRAME # %d: ' % frame_id, '=' * 15)
        # # 输出tracking_list的标注结果
        # for obj in tracking_list:
        #     # 转化为真实距离坐标系的距离尺度 并捡调偏移
        #     print('Corner1-4: ', obj[8:16].reshape(-1, 2) / voxel_scale - cur_car_position)
        #     print('Speed: ', np.sqrt(obj[4] ** 2 + obj[5] ** 2) / voxel_scale)
        #     print('Alt: ', np.arctan(obj[4] / (obj[5] + 1e-10)))

        # 清除队列
        ClusterDetector.PC_QUEUE.clear()

        # 将结果处理为需要的格式n x [4个角坐标,速度,角度]
        n_objs = tracking_list.shape[0]
        corners = tracking_list[:, 8:16].reshape(n_objs, -1, 2) / ClusterDetector.voxel_scale - cur_car_position
        speeds = np.sqrt(tracking_list[:, 4] ** 2 + tracking_list[:, 5] ** 2) / ClusterDetector.voxel_scale
        alts = np.arctan(tracking_list[:, 4] / (tracking_list[:, 5] + 1e-10))
        results = np.hstack((corners.reshape(n_objs, -1), speeds.reshape(n_objs, -1), alts.reshape(n_objs, -1)))
        return tracking_list, results


if __name__ == '__main__':
    # 追踪并可视化
    for seq_id in range(1, 31):
        # 初始化检测类
        detector = ClusterDetector()
        ClusterDetector.VISUALIZE = True
        # 初始化imu数据
        imu_path = './data/first/imuseq/%d' % seq_id

        velo_path = './data/first/velo/%d' % seq_id

        # 上一帧的目标列表
        last_frame_tracking_list = []
        # 计算多帧的聚类结果
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cmp)):
            frame_id = int(frame_file.replace('.bin', ''))
            if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + ClusterDetector.OVERLAP_FRAME_COUNT))):
                break
            # 读取点云数据
            velo_data = load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))
            # 读取imu数据
            imu_data = next(
                csv.reader(open(os.path.join(imu_path, '%d.txt' % frame_id), 'r'), delimiter=' ')
            )
            start = time.clock()
            # 模拟更新追踪结果
            # 保存为下一阵的前驱结果
            last_frame_tracking_list, results = detector.update(
                velo_data,
                imu_data,
                last_frame_tracking_list,
                frame_id
            )
            elapsed = (time.clock() - start)
            print("Time used:", elapsed)
            if ClusterDetector.VISUALIZE:
                # 可视化结果
                detector.visualize_result(last_frame_tracking_list, frame_id, seq_id)
            # print(results)
