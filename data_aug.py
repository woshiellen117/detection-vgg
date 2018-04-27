
import tensorflow as tf
import cv2
import numpy as np
# file_path = "/Volumes/TOSHIBA EXT/final/useful_data_cut_5709-7575"
file_path = "/Volumes/TOSHIBA EXT/final/useful_data_cut_3744-5708"
str = file_path + '/4407.png'
img = cv2.imread(str, 0)

# Flipped Horizontally 水平翻转
h_flip = cv2.flip(img, 1)

# Flipped Vertically 垂直翻转
v_flip = cv2.flip(img, 0)

res1 = np.uint8(np.clip((1.5 * img - 10), 0, 255))#
res1 = cv2.resize(res1,(448,448))
res2 = np.uint8(np.clip((1.8 * img + 15), 0, 255))#
res2 = cv2.resize(res2,(448,448))
res3 = np.uint8(np.clip((1.5 * h_flip - 10), 0, 255))#
res3 = cv2.resize(res3,(448,448))
res4 = np.uint8(np.clip((1.8 * h_flip + 15), 0, 255))#
res4 = cv2.resize(res4,(448,448))
res5 = np.uint8(np.clip((1.5 * v_flip - 10), 0, 255))#
res5 = cv2.resize(res5,(448,448))
res6 = np.uint8(np.clip((1.8 * v_flip + 15), 0, 255))#
res6 = cv2.resize(res6,(448,448))
img = cv2.resize(img,(448,448))
tmp = np.hstack((img, res1))
tmp = np.hstack((tmp, res2))
tmp = np.hstack((tmp, res3))
tmp = np.hstack((tmp, res4))
tmp = np.hstack((tmp, res5))
tmp = np.hstack((tmp, res6))

res = cv2.resize(img,(448,448))

cv2.imwrite('tmp.jpg',tmp)
cv2.imwrite('res.jpg',res)