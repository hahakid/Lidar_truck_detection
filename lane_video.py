import functools
import os
import cv2
import time


def cmp_file(a: str, b: str):
    """
    文件名的纯数字排序比较函数
    :param a:
    :param b:
    :return:
    """
    file_number_a = int(a[:a.rfind('.')])
    file_number_b = int(b[:b.rfind('.')])
    if file_number_a > file_number_b:
        return 1
    else:
        return -1


base_dir = './output/4'
file_list = os.listdir(base_dir)
file_list = sorted(file_list * 3, key=functools.cmp_to_key(cmp_file))
fps = 24
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    'cluster_4.mp4',
    fourcc,
    fps,
    (1855, 1006)
)
for img_file in file_list:
    print(img_file)
    img = cv2.imread(os.path.join(base_dir, img_file))
    video.write(img)
video.release()
