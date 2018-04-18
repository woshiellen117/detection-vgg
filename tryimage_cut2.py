
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
str = file_path + '/6042.png'
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

x = np.arange(1, image.shape[1]+1, 1)
y = np.array(w)
#第一个拟合，自由度为3
z1 = np.polyfit(x, y, 18)
# 生成多项式对象
p1 = np.poly1d(z1)
print(z1)
print(p1)

pl.plot(x, y, 'b^-', label='Origin Line')
pl.plot(x, p1(x), 'gv--', label='Poly Fitting Line(deg=18)')
pl.axis([0, 1000, -500, 3000])
pl.legend()
# Save figure
pl.savefig('scipy02.png', dpi=96)

d1=np.polyder(z1)#求多项式的导数系数
dhs1=np.poly1d(d1)#求导函数
root = np.roots(dhs1)
print(root)
j=0
root1=[]
for i in range(root.size):
    if root[i].imag==0 and root[i].real>0 and root[i]<image.shape[1]:
        root1.append(root[i])
        j=j+1
print(root1)
middle=image.shape[1]/2
for i in range(j):
    if root1[i]>0 and root1[i]<=middle:
        i1=i
        if p1(root1[i])<p1(root1[i-1]):
            while i-2>=0 and abs(p1(root1[i-1])-p1(root1[i-2])<=300) and p1(root1[i-1])<=350:
                right=root1[i-2]
                i=i-1
            if i-2<=0:
                right=root1[0]
            elif p1(root[i-1])>350:
                right=root1[i-1]
            elif abs(p1(root1[i-1])-p1(root1[i-2]))>300:
                right=root1[i-1]
            i=i1
            while i+2<j and abs(p1(root1[i+1])-p1(root1[i+2]))<=300 and p1(root1[i+1])<=350:
                left=root1[i+2]
                i=i+1
            if i+2>=j:
                left=root1[j-1]
            elif p1(root1[i+1])>350:
                left = root1[i + 1]
            elif abs(p1(root1[i + 1]) - p1(root1[i + 2])) > 300:
                left=root1[i+1]
            break
        elif p1(root1[i])>=p1(root1[i-1]):
            while i-3>=0 and abs(p1(root1[i-2]-p1(root1[i-3])<=300)) and p1(root1[i-2])<=350:
                right=root1[i-3]
                i=i-1
            if i-3<0:
                right=root1[0]
            elif p1(root1[i-2])>350:
                right=root1[i-2]
            elif abs(p1(root1[i - 2] - p1(root1[i - 3])) > 300):
                right=root1[i-2]
            i=i1
            while i+1<j and abs(p1(root1[i])-p1(root1[i+1]))<=300 and p1(root1[i])<=350:
                left=root1[i+1]
                i=i+1
            if i+1>=j:
                left==root1[j-1]
            elif p1(root1[i])>350:
                left=root1[i]
            elif abs(p1(root1[i]) - p1(root1[i + 1])) > 300:
                left=root1[i]
            break

print(left)
print(right)

cut = np.zeros((image.shape[0],int(right)-int(left)+1), np.uint8)
cut_x=0
for x in range(int(left), int(right)+1):
    for y in range(image.shape[0]):
        cut[y][cut_x] = image[y][x]
    cut_x=cut_x+1

cv2.imwrite('image.png',paintx)
cv2.imwrite('6042.png',cut)
