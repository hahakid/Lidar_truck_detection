import functools
import os
import cv2
import time
import pylab as plt
from clyent import color


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


fps = 24
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    'cluster_all.mp4',
    fourcc,
    fps,
    (1855, 1006)
)
for i in range(1, 26):
    base_dir = './output/%d' % i
    file_list = os.listdir(base_dir)
    file_list = sorted(file_list * 3, key=functools.cmp_to_key(cmp_file))
    for img_file in file_list:
        print(img_file)
        img = cv2.imread(os.path.join(base_dir, img_file))
        cv2.putText(
            img,
            text='SEQ %d' % (i),
            org=(40, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            lineType=2
        )
        video.write(img)

video.release()
