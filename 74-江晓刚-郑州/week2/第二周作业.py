# -*- coding: utf-8 -*-
# @Time : 2022/11/8 19:30
# @Author : 江晓刚
# @File : 第二周作业.py
# @Software: PyCharm


from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 灰度化
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #转换BRG为RGB  opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！
h,w = img.shape[:2]  # 获取图片的长（h）和宽（w）
img_gray = np.zeros([h,w],img.dtype) # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j] # 取出当前长和款中的RGB坐标
        img_gray[i,j] = int(m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11) # 将RGB坐标转化为gray坐标并赋值给新图像
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)
cv2.imwrite('./h_img.jpg',img_gray)    #imwrite()函数用来保存图片

plt.subplot(221)
# 使用plt.subplot来创建小图. plt.subplot(221)表示将整个图像窗口分为2行2列, 当前位置为1.
# plt.subplot(221)表示将整个图像窗口分为2行2列, 当前位置为1.
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)


img_gray = rgb2gray(img)
plt.subplot(222)
# plt.subplot(222)表示将整个图像窗口分为2行2列, 当前位置为2.
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)


"""
imshow函数详解
# 对于imshow函数，opencv的官方注释指出：根据图像的深度，imshow函数会自动对其显示
# 灰度值进行缩放，规则如下：
#1：如果图像数据类型是8U（8位无符号），则直接显示。
#2：:如果图像数据类型是16U（16位无符号）或32S（32位有符号整数），则imshow函数内部
# 会自动将每个像素值除以256并显示，即将原图像素值的范围由[0~255*256]映射到[0~255]
#3：如果图像数据类型是32F（32位浮点数）或64F（64位浮点数），则imshow函数内部会自动
# 将每个像素值乘以255并显示，即将原图像素值的范围由[0~1]映射到[0~255]（注意：原图
# 像素值必须要归一化
"""

# 通过cvtColor函数实现灰度化
#img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
#plt.subplot(222)
#plt.imshow(img_gray, cmap='gray')
#print("---image gray----")
#print(img_gray)

# 二值化
rows,cols = img_gray.shape
for i in range(rows):
    for j in  range(cols):
        if (img_gray[i,j] <= 0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
plt.subplot(223)
# plt.subplot(223)表示将整个图像窗口分为2行2列, 当前位置为3.
plt.imshow(img_gray, cmap='gray')

# plt.imshow()函数中的cmap = 'gray'是以灰度图形式展示的意思

# 函数实现灰度化
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(224)
# plt.subplot(224)表示将整个图像窗口分为2行2列, 当前位置为4.
plt.imshow(img_gray2, cmap='gray')
plt.show()
