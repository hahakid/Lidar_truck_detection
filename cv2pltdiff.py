import cv2
import os
import numpy
import matplotlib.pyplot as plt

path_1c=r'F:\daxie\daxie\differ\1'
path_3c=r'F:\daxie\daxie\differ\3'
numpy.set_printoptions(precision=3)
data_1c_cv2=cv2.imread(os.path.join(path_1c,'0.png'))
#data_1c_cv2=plt.imread(os.path.join(path_1c,'0.png'))
shape=data_1c_cv2.shape
if len(shape)==2:
    numpy.savetxt('../differ/data_1c_cv2_%d.csv' % 0, data_1c_cv2, fmt='%d', delimiter=' ')
else:
    for i in range(0,shape[-1]):
        numpy.savetxt('../differ/data_1c_cv2_%d.csv' %i, data_1c_cv2[:,:,i], fmt='%d', delimiter=' ')

#data_1c_plt=cv2.imread(os.path.join(path_1c,'_0.png'))
data_1c_plt=plt.imread(os.path.join(path_1c,'_0.png'))
shape=data_1c_plt.shape
if len(shape)==2:
    numpy.savetxt('../differ/data_1c_plt_%d.csv' % 0, data_1c_plt[:, :, i], fmt='%f', delimiter=' ')
else:
    for i in range(0,shape[-1]):
        numpy.savetxt('../differ/data_1c_plt_%d.csv' %i, data_1c_plt[:,:,i], fmt='%f',delimiter=' ')

#data_3c_cv2=cv2.imread(os.path.join(path_3c,'0.png'))
data_3c_cv2=plt.imread(os.path.join(path_3c,'0.png'))

shape=data_3c_cv2.shape
if len(shape)==2:
    numpy.savetxt('../differ/data_3c_cv2_%d.csv' % 0, data_3c_cv2[:, :, i], fmt='%f', delimiter=' ')
else:
    for i in range(0,shape[-1]):
        numpy.savetxt('../differ/data_3c_cv2_%d.csv' %i, data_3c_cv2[:,:,i], fmt='%f', delimiter=' ')

#data_3c_plt=cv2.imread(os.path.join(path_3c,'_0.png'))
data_3c_plt=plt.imread(os.path.join(path_3c,'_0.png'))
shape=data_3c_plt.shape
if len(shape)==2:
    numpy.savetxt('../differ/data_3c_plt_%d.csv' %0, data_3c_plt[:,:,i], fmt='%f', delimiter=' ')
else:
    for i in range(0,shape[-1]):
        numpy.savetxt('../differ/data_3c_plt_%d.csv' %i, data_3c_plt[:,:,i], fmt='%f', delimiter=' ')

