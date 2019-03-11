#_*_ coding=utf-8 _*_
import functools
import os
import glob
import cv2
import time
import pylab as plt
from clyent import color

width=480
height=480
fps=10

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

def readlabel(path):
    label=path.replace('png','txt')
    f = open(label)
    lines = f.readlines()
    f.close()
    labels=[]
    for l in lines:
        l = l.split(" ")
        xmid = float(l[1])
        ymid = float(l[2])
        bwidth = float(l[3])
        bheight = float(l[4])
        xmin = int((xmid - bwidth / 2) * width)
        ymin = int((ymid - bheight / 2) * height)
        xmax = int((xmid + bwidth / 2) * width)
        ymax = int((ymid + bheight / 2) * height)
        labels.append([xmin,ymin,xmax,ymax])
    return labels
    
            
def to_video(imglist,base_dir,outpath,idx):
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path=os.path.join(outpath,'%d.mp4' %idx)
    video = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (width, height)
    )
    for img_file in imglist:
        labels=readlabel(os.path.join(base_dir, img_file))
        
        img = cv2.imread(os.path.join(base_dir, img_file))
        '''
        cv2.putText(
            img,
            text='GT(red)',
            org=(10, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            lineType=2
        )
        cv2.putText(
            img,
            text='DT(green)',
            org=(10, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            lineType=2
        )
        '''
        if len(labels)>0:
            for l in labels:
                img=cv2.rectangle(img,(l[0],l[1]),(l[2],l[3]),(0,0,255),1) #bgr???
        video.write(img)

    video.release()

    
def main():
    father_path='/media/kid/workspace/data/daxie/2/'
    channel='birdview/feature_4c'
    for i in range(0,33):
        img_path=os.path.join(father_path,channel,str(i))
        file_list=os.listdir(img_path)
        im_list=[]
        for j in file_list:
            if not 'txt' in j:
                im_list.append(j)
        #print(im_list)
        im_list = sorted(im_list , key=functools.cmp_to_key(cmp_file))
        #print(im_list)
        output_path=os.path.join(father_path,'video/feature_4c','raw')
        #print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        to_video(im_list,img_path,output_path,i)
        
main()
