
import tensorflow as tf
import cv2
import numpy as np
import pylab as pl
import skimage.io as io
import xlrd
SERIAL_NUM = 0
NUM_PIC = 13768
IMAGE_SIZE = 700
#
# # 灰度化读取图片
# file_path = "/Volumes/TOSHIBA EXT/final/useful_data"
# str = file_path + '/16633.png'
# image = cv2.imread(str, 0)
def load_name(xml_path):
    labels_xml = xlrd.open_workbook(xml_path)
    labels_table = labels_xml.sheets()[0]
    serial_number = labels_table.col_values(SERIAL_NUM)[1:NUM_PIC+1]
    return serial_number

file_path = "/Volumes/TOSHIBA EXT/final/useful_data"
str1= file_path + '/*.png'
# str1 = file_path + '/0.png'
coll = io.ImageCollection(str1)

xml_path = "/Volumes/TOSHIBA EXT/final/useful_data/useful_data.xls"
name = load_name(xml_path)

for mm in range(NUM_PIC):
    print('No.')
    print(mm)
    image = cv2.resize(coll[mm], (IMAGE_SIZE, coll[mm].shape[1]), interpolation=cv2.INTER_CUBIC)
    # 创建一个空白图片(img.shape[0]为height,img.shape[1]为width)
    paintx = np.zeros(image.shape, np.uint8)

    # 将新图像数组中的所有通道元素的值都设置为0
    # cv2.cv.Zero(cv2.cv.fromarray(paintx))

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
                if i - 2 >= 0:
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

    cut = np.zeros((coll[mm].shape[0],int(right)-int(left)+1), np.uint8)
    cut_x=0
    for x in range(int(left), int(right)+1):
        for y in range(coll[mm].shape[0]):
            cut[y][cut_x] = coll[mm][y][x]
        cut_x=cut_x+1
    print(str(name[mm]))
    print(name[mm])
    cv2.imwrite('/Volumes/TOSHIBA EXT/final/useful_data_cut_1_func/'+str(int(name[mm]))+'.jpg', paintx)
    cv2.imwrite('/Volumes/TOSHIBA EXT/final/useful_data_cut_1/'+str(int(name[mm]))+'.jpg', cut)