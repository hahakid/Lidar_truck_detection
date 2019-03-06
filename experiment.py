import csv
import functools
import os
import shutil
import numpy as np
import time

import cv2

import pandas as pd
import pylab as plt
import mayavi
from mayavi import mlab
from cluster_algo import ClusterDetector
from functions import utils, feature_projection


def mkdirs():
    """
    初始画文件夹
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def process_frame(velo_data, imu_data):
    # 加入队列 满足叠加帧数则叠加
    detector.PC_QUEUE.append([velo_data, imu_data])
    if len(detector.PC_QUEUE) < ClusterDetector.OVERLAP_FRAME_COUNT:
        return None
    else:
        # 如果达到了要求叠加的帧数 则处理
        voxel, imu = detector.overlap_voxelization(
            point_cloud_id=0,
            overlap_frame_count=ClusterDetector.OVERLAP_FRAME_COUNT
        )

        detector.PC_QUEUE.clear()

        # 点云逆变换到车体坐标
        abs_x, abs_y, heading = imu
        voxel[:, 0] = voxel[:, 0] - abs_x
        voxel[:, 1] = voxel[:, 1] - abs_y
        voxel[:, :3] = detector.rotate(voxel[:, :3], -heading)
        # 修正点云变换造成的误差
        voxel[:, 1] -= 0.5
        # 再进行一次过滤 过滤掉叠加之后 在车辆前进方向上多出来的点
        voxel = detector.passthrough_filter(voxel).to_array()
        if len(voxel) == 0: return None

        # 边界要和重叠前点云的边界一致
        boundary = {
            'minX': detector.BORDER_TH[0],
            'maxX': detector.BORDER_TH[3],
            'minY': detector.BORDER_TH[1],
            'maxY': detector.BORDER_TH[4],
            'minZ': detector.BORDER_TH[2],
            'maxZ': detector.BORDER_TH[5],
        }

        feature_map = feature_projection.makeBVFeature(
            voxel,
            boundary,
            Discretization=512
        )
        return feature_map


if __name__ == '__main__':
    # 追踪并可视化
    for seq_id in range(1, 31):
        father_path = '/media/hviktortsoi/D/dataset/Radar201901/Data2019/first'
        feature_type = 'feature_3c'
        save_path = './output/%s/%d' % (feature_type, seq_id)
        # 创建文件夹
        mkdirs()

        # 初始化imu数据
        imu_path = './data/first/imuseq/%d' % seq_id
        velo_path = './data/first/velo/%d' % seq_id

        # 初始化检测类
        ClusterDetector.VISUALIZE = False
        ClusterDetector.IS_FILTER_GROUND = True
        ClusterDetector.OVERLAP_FRAME_COUNT = 2
        ClusterDetector.VOXEL_GRID_SIZE = 0.1
        ClusterDetector.BORDER_TH = [-30, -10, -1.9, 30, 50, 2.6]
        ClusterDetector.GROUND_LIMIT = (0.5, 5)
        detector = ClusterDetector()

        # 读取点云
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(utils.cmp)):
            frame_id = int(frame_file.replace('.bin', ''))
            if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + ClusterDetector.OVERLAP_FRAME_COUNT))):
                break
            # 读取点云数据
            velo_data = utils.load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))
            # 读取imu数据
            imu_data = next(
                csv.reader(open(os.path.join(imu_path, '%d.txt' % frame_id), 'r'), delimiter=' ')
            )
            # 处理当前帧数据
            feature_map = process_frame(velo_data, imu_data)
            if isinstance(feature_map, np.ndarray):
                # 转置 这里要用copy
                # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
                feature_map = np.rot90(feature_map).copy()
                # 预处理图像
                # 膨胀
                kernel = np.ones((3, 3), np.uint8)
                feature_map = cv2.dilate(feature_map, kernel, iterations=3)
                # 保存图像
                # cv2.imwrite(
                #     save_path + '/%d.jpg' % frame_id,
                #     feature_map * 255,
                #     [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                # )
                # shutil.copy(os.path.join(father_path, 'label', '%d' % seq_id, '%d.txt' % frame_id), save_path)
                # 读取标签
                try:
                    labels = pd.read_csv(
                        os.path.join(father_path, 'label', '%d' % seq_id, '%d.txt' % frame_id),
                        header=None, sep=' '
                    ).values
                except Exception as e:
                    continue
                for label in labels:
                    c, x, y, w, h = \
                        label[0], \
                        label[1] * feature_map.shape[1], \
                        label[2] * feature_map.shape[0], \
                        label[3] * feature_map.shape[1], \
                        label[4] * feature_map.shape[0]
                    print(label)
                    # 绘制标签
                    cv2.rectangle(
                        feature_map,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        color=(255, 255, 255), thickness=1
                    )
                # 可视化图像
                cv2.imshow('', feature_map)
                cv2.waitKey(0)
