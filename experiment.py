import csv
import functools
import os
import time

import pylab as plt
import mayavi
from mayavi import mlab
from cluster_algo import ClusterDetector
from functions import utils, feature_projection

if __name__ == '__main__':
    fig = mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640 * 2, 360 * 2))
    boundary = {'minX': -30, 'maxX': 30, 'minY': -10, 'maxY': 50, 'minZ': -1.6, 'maxZ': 2.4, }
    pixel = 512  # pixel 480=32*15(yolo)=60*8
    # 追踪并可视化
    for seq_id in range(1, 31):
        # 初始化检测类
        ClusterDetector.VISUALIZE = False
        detector = ClusterDetector()
        # 初始化imu数据
        imu_path = './data/first/imuseq/%d' % seq_id

        velo_path = './data/first/velo/%d' % seq_id

        # 上一帧的目标列表
        last_frame_tracking_list = []
        overlapped_frame_count = 0
        # 计算多帧的聚类结果
        for frame_file in sorted(os.listdir(velo_path), key=functools.cmp_to_key(utils.cmp)):
            frame_id = int(frame_file.replace('.bin', ''))
            if not os.path.exists(os.path.join(velo_path, '%d.bin' % (frame_id + ClusterDetector.OVERLAP_FRAME_COUNT))):
                break
            # 读取点云数据
            velo_data = utils.load_velo_scan(os.path.join(velo_path, '%d.bin' % frame_id))[:, :3]
            # 读取imu数据
            imu_data = next(
                csv.reader(open(os.path.join(imu_path, '%d.txt' % frame_id), 'r'), delimiter=' ')
            )
            # 加入队列 满足叠加帧数则叠加
            if overlapped_frame_count < ClusterDetector.OVERLAP_FRAME_COUNT:
                detector.PC_QUEUE.append([velo_data, imu_data])
                overlapped_frame_count += 1
            else:
                # 如果达到了要求叠加的帧数 则处理
                voxel, imu_data = detector.overlap_voxelization(
                    point_cloud_id=0,
                    overlap_frame_count=ClusterDetector.OVERLAP_FRAME_COUNT
                )
                overlapped_frame_count = 0
                detector.PC_QUEUE.clear()
                # mayavi.mlab.points3d(
                #     voxel[:, 0], voxel[:, 1], voxel[:, 2],
                #     # clustering.labels_,
                #     # mode="cube",
                #     mode="point",
                #     # color=(0, 1, 0),
                #     colormap='spectral',
                #     figure=fig,
                #     scale_factor=1
                # )
                # # mlab.outline()
                # # xy平面0度 z轴0度 摄像机离xy平面300米 焦点在整个图形中点
                # mlab.view(0, 30, 300, focalpoint=(0, 0, 0))
                # mlab.savefig(filename='/tmp/tmp.png', figure=fig)
                # mlab.clf()
                feature_map = feature_projection.makeBVFeature(
                    overlapped_frame_count,
                    boundary,
                    pixel
                )
                plt.imshow(feature_map)
                plt.show()

                time.sleep(1)
