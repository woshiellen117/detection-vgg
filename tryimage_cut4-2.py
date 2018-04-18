
import tensorflow as tf
import cv2
import numpy as np
import pylab as pl
import skimage.io as io
import xlrd
import os
import glob as gb
SERIAL_NUM = 0

img_path = gb.glob("/Volumes/TOSHIBA EXT/final/useful_data_7576-8955/*.png")
# file_path = "D:/final/useful_data_1924-3743"
for path in img_path:
    image = cv2.imread(path,0)

    # 创建width长度都为0的数组
    w = [0] * image.shape[1]
    # 对每一行计算投影值
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            t = image[y][x]
            if t < 40:
                w[x] += 1



    x = np.arange(1, image.shape[1] + 1, 1)
    y = np.array(w)
    # 第一个拟合，自由度为3
    z1 = np.polyfit(x, y, 18)
    # 生成多项式对象
    p1 = np.poly1d(z1)
    print(z1)
    print(p1)


    d1 = np.polyder(z1)  # 求多项式的导数系数
    dhs1 = np.poly1d(d1)  # 求导函数
    root = np.roots(dhs1)
    print(root)
    j = 0
    root1 = []
    for i in range(root.size):
        if root[i].imag == 0 and root[i].real > 0 and root[i] < image.shape[1]:
            root1.append(root[i])
            j = j + 1
    print(root1)
    middle = image.shape[1] / 2

    for i in range(j):
        if root1[i] > 0 and root1[i] <= middle:
            i1 = i
            if p1(root1[i]) < p1(root1[i - 1]):
                while i - 2 >= 0 and abs(p1(root1[i - 1]) - p1(root1[i - 2]) <= 300) and p1(root1[i - 1]) <= 350:
                    right = root1[i - 2]
                    i = i - 1
                if i - 2 <= 0:
                    right = root1[0]
                elif p1(root[i - 1]) > 350:
                    right = root1[i - 1]
                elif abs(p1(root1[i - 1]) - p1(root1[i - 2])) > 300:
                    right = root1[i - 1]
                i = i1
                while i + 2 < j and abs(p1(root1[i + 1]) - p1(root1[i + 2])) <= 300 and p1(root1[i + 1]) <= 350:
                    left = root1[i + 2]
                    i = i + 1
                if i + 2 >= j:
                    left = root1[j - 1]
                elif p1(root1[i + 1]) > 350:
                    left = root1[i + 1]
                elif abs(p1(root1[i + 1]) - p1(root1[i + 2])) > 300:
                    left = root1[i + 1]
                break
            elif p1(root1[i]) >= p1(root1[i - 1]):
                while i - 3 >= 0 and abs(p1(root1[i - 2] - p1(root1[i - 3]) <= 300)) and p1(root1[i - 2]) <= 350:
                    right = root1[i - 3]
                    i = i - 1
                if i - 3 < 0:
                    right = root1[0]
                elif p1(root1[i - 2]) > 350:
                    right = root1[i - 2]
                elif abs(p1(root1[i - 2] - p1(root1[i - 3])) > 300):
                    right = root1[i - 2]
                i = i1
                while i + 1 < j and abs(p1(root1[i]) - p1(root1[i + 1])) <= 300 and p1(root1[i]) <= 350:
                    left = root1[i + 1]
                    i = i + 1
                if i + 1 >= j:
                    left == root1[j - 1]
                elif p1(root1[i]) > 350:
                    left = root1[i]
                elif abs(p1(root1[i]) - p1(root1[i + 1])) > 300:
                    left = root1[i]
                break

    print(int(left))
    print(int(right))
    if int(left)<0:
        left=0
    elif int(right)>image.shape[1]-1:
        right = image.shape[1]-1
    #     cut = np.zeros((image.shape[0], int(right) - int(left) + 1), np.uint8)
    # cut_x=0
    # for x in range(int(left), int(right)+1):
    #     for y in range(image.shape[0]):
    #         cut[y][cut_x] = image[y][x]
    #     cut_x=cut_x+1
    cut=image[0:image.shape[0],int(left):int(right)]
    cv2.imwrite(path[0:39] + "cut_" + path[39:], cut)