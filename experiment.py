import csv
import functools
import os
import time

import mayavi
from mayavi import mlab
import cluster_algo
from functions import funtiontest

for seq_id in range(1, 31):

    # 真实世界尺度(米)对应的voxel格子数
    # voxel_scale = VOXEL_GRANULARITY / (BORDER_TH[3] - BORDER_TH[0])
    voxel_scale = 1

    ORIGIN = None
    # 初始化imu数据
    imu_path = './data/first/imuseq/%d' % seq_id
    # imu_data_processer = get_imu_data()  # imu数据处理器
    # next(imu_data_processer)

    velo_path = './data/first//velo/%d' % seq_id

    # 上一帧的目标列表
    last_frame_tracking_list = []
    # 初始化绘图的figure
    cluster_algo.fig = \
        mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640 * 2, 360 * 2)) if cluster_algo.VISUALIZE else None
    # 计算多帧的聚类结果
    for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(cluster_algo.cmp)):
        frame_id = int(frame_file.replace('.bin', ''))
        if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + cluster_algo.OVERLAP_FRAME_COUNT))):
            break
        # 读取点云数据
        velo_data = funtiontest.load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))[:, :3]
        # 读取imu数据
        imu_data = next(
            csv.reader(open(os.path.join(imu_path, '%d.txt' % frame_id), 'r'), delimiter=' ')
        )
        start = time.clock()
        # 模拟更新追踪结果
        # 保存为下一阵的前驱结果
        last_frame_tracking_list, results = cluster_algo.update(velo_data, imu_data, last_frame_tracking_list, frame_id)
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
