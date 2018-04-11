
import tensorflow as tf
import cv2
import numpy as np
import pylab as pl

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
str = file_path + '/393.png'
image = cv2.imread(str, 0)
# cv2.imshow('aa',image)
# cv2.waitKey(0)
# 将图片二值化

# retval,img = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
# cv2.imshow('aa',img)
# cv2.waitKey(0)

# 创建一个空白图片(img.shape[0]为height,img.shape[1]为width)
paintx = np.zeros(image.shape, np.uint8)

# 将新图像数组中的所有通道元素的值都设置为0
# cv2.cv.Zero(cv2.cv.fromarray(paintx))

# 创建width长度都为0的数组
w = [0] * image.shape[1]
for x in range(image.shape[0]):
    print(image[x][350])
# 对每一行计算投影值
for x in range(image.shape[1]):
    for y in range(image.shape[0]):
        t = image[y][x]
        if t<40:
            w[x] += 1

# 绘制垂直投影图
for x in range(image.shape[1]):
    for y in range(w[x]):
        paintx[y,x]=255#wight
    for y in range(w[x],paintx.shape[0]):
        paintx[y,x]=0;#black
print(w)

x = np.arange(1, image.shape[1]+1, 1)
y = np.array(w)
#第一个拟合，自由度为3
z1 = np.polyfit(x, y, 3)
# 生成多项式对象
p1 = np.poly1d(z1)
print(z1)
print(p1)
# 第二个拟合，自由度为6
z2 = np.polyfit(x, y, 6)
# 生成多项式对象
p2 = np.poly1d(z2)
print(z2)
print(p2)
# 绘制曲线
#  原曲线
pl.plot(x, y, 'b^-', label='Origin Line')
pl.plot(x, p1(x), 'gv--', label='Poly Fitting Line(deg=3)')
pl.plot(x, p2(x), 'r*', label='Poly Fitting Line(deg=6)')
pl.axis([0, 1000, 0, 3000])
pl.legend()
# Save figure
pl.savefig('scipy02.png', dpi=96)

# 显示图片
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',paintx)
cv2.imwrite('image.jpg',paintx)
cv2.waitKey(0)