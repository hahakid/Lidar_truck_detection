import functools
import numpy as np
import os
from functions import top_view
import matplotlib.pyplot as plt
import struct
from mayavi import mlab
import mayavi
from PointcloudVoxelizer.source import pointclouds_to_voxelgrid
from sklearn.cluster import DBSCAN

border_thresh = (-30, -20, -1.6, 30, 40, 2.4)


def cmp(a, b):
    l, r = int(a.replace('.bin', '')), int(b.replace('.bin', ''))
    return -1 if l < r else 1 if l > r else 0


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def load_velo_scan_v0(velo_filename):
    bin_file = open(velo_filename, 'rb')
    # 读取个数
    count = int.from_bytes(bin_file.read(4), byteorder='little')
    # 提取x y z
    x = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    y = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    z = [struct.unpack('f', bin_file.read(4))[0] for _ in range(count)]
    reflect = [struct.unpack('B', bin_file.read(1))[0] for _ in range(count)]
    rgb_filler = [0 for _ in range(len(x))]
    scan = np.array([x, y, z, rgb_filler, rgb_filler, rgb_filler, reflect]).squeeze().T
    return scan


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


@mlab.animate(delay=50)
def anim():
    plt = None
    for sub in l:
        velo0 = load_velo_scan_v0(os.path.join(velo_path0, sub))
        velo1 = load_velo_scan_v0(os.path.join(velo_path1, sub))
        velo = np.vstack((velo0, velo1))
        voxel_converter = pointclouds_to_voxelgrid.data_loader(point_list=velo)
        full_mat, threeD_mat, threeD_label, list_mat, label_mat = voxel_converter(
            xyz_range=border_thresh,
            mag_coeff=300
        )

        xx, yy, zz = np.where(threeD_label > 0)
        # 取定长数组构成图像
        max_len = 4000
        data_len = xx.shape[0]
        xx = np.hstack([xx.reshape(1, -1), np.zeros([1, max_len - data_len])]).reshape(-1)
        yy = np.hstack([yy.reshape(1, -1), np.zeros([1, max_len - data_len])]).reshape(-1)
        zz = np.hstack([zz.reshape(1, -1), np.zeros([1, max_len - data_len])]).reshape(-1)
        # 聚类
        clustering = DBSCAN(eps=1, min_samples=3).fit(
            np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
        )
        # fig = mlab.gcf()
        fig = mayavi.mlab.figure(mlab.gcf(), bgcolor=(0, 0, 0), size=(640, 360))
        if plt is None:
            plt = mayavi.mlab.points3d(
                xx, yy, zz,
                clustering.labels_,  # Values used for Color
                # mode="cube",
                mode="point",
                # color=(0, 1, 0),
                colormap='spectral',
                figure=fig,
                scale_factor=1
            )
        else:
            plt.mlab_source.set(x=xx, y=yy, z=zz, s=clustering.labels_)
        yield


def gen_birdview_pics():
    pic_path = './output/'
    l = sorted(os.listdir(velo_path0))
    for sub in l:
        velo0 = load_velo_scan_v0(os.path.join(velo_path0, sub))
        velo1 = load_velo_scan_v0(os.path.join(velo_path1, sub))
        velo = np.vstack((velo0, velo1))
        # print velo.shape
        # velo=velo.T
        # print velo.shape

        # '''
        # img = panoramic.point_cloud_to_panorama(velo)
        # img=panoramic.point_cloud_to_panorama(velo,v_res=1.33,h_res=0.4,v_fov=(-30.67,10.67))# 32
        img = top_view.point_cloud_2_birdseye(velo, res=0.1,
                                              side_range=(-30., 30.),  # left-most to right-most
                                              fwd_range=(-20., 40.),  # back-most to forward-most
                                              height_range=(-1.6, 2.4))
        plt.figure(figsize=(9.6, 16.8))  # 4:7
        plt.imshow(img)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        framename = pic_path + sub.split('.')[0] + '.png'
        plt.savefig(framename, dpi=100)
        plt.close()
        # '''


if __name__ == '__main__':
    velo_path0 = './data/Data-2019-01-21-23-28-32/VLP160'
    velo_path1 = './data/Data-2019-01-21-23-28-32/VLP161'
    # 聚类
    velo0 = load_velo_scan(os.path.join(velo_path0, '230.bin'))
    velo1 = load_velo_scan(os.path.join(velo_path1, '230.bin'))
    velo = np.vstack((velo0, velo1))
    voxel_converter = pointclouds_to_voxelgrid.data_loader(point_list=velo)
    full_mat, threeD_mat, threeD_label, list_mat, label_mat = voxel_converter(
        xyz_range=border_thresh,
        mag_coeff=300
    )
    xx, yy, zz = np.where(threeD_label > 0)
    clustering = DBSCAN(eps=1, min_samples=4, n_jobs=-1).fit(
        np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
    )
    print(clustering.labels_)
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(xx, yy, zz,
                         # clustering.labels_,  # Values used for Color
                         mode="cube",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         scale_factor=2,
                         )
    mayavi.mlab.show()
    exit(0)
    l = sorted(os.listdir(velo_path0), key=functools.cmp_to_key(cmp))
    print('总长度', len(l))
    anim()
    mlab.show()
