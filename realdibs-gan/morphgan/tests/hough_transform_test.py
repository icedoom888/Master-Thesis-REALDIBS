import cv2
import numpy as np
import tensorflow as tf


in_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/in_imgs/'
tar_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/tar_imgs/'

'''
in_test = np.load(in_path + 'in_test.npy')
tar_test = np.load(tar_path + 'tar_test.npy')

in_test = (in_test * 0.5 + 0.5)*255
tar_test = (tar_test * 0.5 + 0.5)*255
'''
inp = np.load(in_path + 'in_test.npy')[0]
inp = np.uint8((inp* 0.5 + 0.5)*255)
gray_in = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)

tar = np.load(tar_path + 'tar_test.npy')[0]
tar = np.uint8((tar* 0.5 + 0.5)*255)
gray_tar = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

edges_in = cv2.Canny(gray_in, 10, 500, apertureSize = 3)
edges_tar = cv2.Canny(gray_tar, 10, 500, apertureSize = 3)

dif = np.abs(edges_in - edges_tar)

while(1):
    cv2.imshow('edges_tar', edges_tar)
    cv2.imshow('edges_in', edges_in)
    cv2.imshow('dif', dif)
    key = cv2.waitKey(0)
    if key == ord('a'):
        break


tf_in = tf.image.sobel_edges(inp)
tf_tar = tf.image.sobel_edges(tar)

import matplotlib.pyplot as plt

plt.imshow(tf_in)
plt.show()
