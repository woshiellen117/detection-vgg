
import tensorflow as tf
import numpy as np
import cv2

file_path = "D:\\final\\useful_data"
str = file_path + '\\16629.png'
image = cv2.imread(str)
image_numpy=np.array(image)
lungwin = np.array([50., 150.])
newimg = (image_numpy - lungwin[0]) / (lungwin[1] - lungwin[0])
newimg[newimg < 0] = 0
newimg[newimg > 1] = 255
cv2.imshow('new',newimg[1000:,:])
cv2.waitKey(0)