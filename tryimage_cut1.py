
import tensorflow as tf
import cv2
import numpy as np


# file_path = "/Volumes/TOSHIBA EXT/final/useful_data"
# str = file_path + '/16629.png'
# image = cv2.imread(str)
# image_numpy=np.array(image)
# lungwin = np.array([50., 150.])
# newimg = (image_numpy - lungwin[0]) / (lungwin[1] - lungwin[0])
# newimg[newimg < 0] = 0
# newimg[newimg > 1] = 255
# cv2.imshow('new',newimg[1000:1500,:])
# cv2.waitKey(0)


# 灰度化读取图片
file_path = "/Volumes/TOSHIBA EXT/final/useful_data"
str = file_path + '/141.png'
image = cv2.imread(str, 0)#351,663  141.png

paintx = np.zeros(image.shape, np.uint8)

for x in range(295,765):
    for y in range(image.shape[0]):
        paintx[y][x] = image[y][x]

cv2.imshow('image',paintx)
cv2.imwrite('8_cut.jpg',paintx)
cv2.waitKey(0)