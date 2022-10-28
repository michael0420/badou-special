##############  1实现RGB2Gray######################

# 用PIL中Image做图片的RGB2GRAY，有噪声

from PIL import Image
image = Image.open('./timg3.jpg')  #打开图片
image = image.convert('1')    #将图片转化为黑白图像
image.save('timg4.jpg')     #保存图片

# 用CV2做图片的RGB2GRAY

import cv2   #导入opencv
img = cv2.imread('timg3.jpg')   #图片导入
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将导入的图片进行灰度化处理
cv2.imwrite('./timg5.jpg',gray)    #imwrite()函数用来保存图片


# ################ 2实现图片的二值化########################
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文

# 读取灰度图像
img = cv2.imread("timg3.jpg",0)
print("原图的shape: ", img.shape)
plt.subplot(2, 2, 1)   #图片的位置
plt.imshow(img, cmap='gray')  #cmap = 'gray'是以灰度图形式展示的意思
plt.title("灰度值图片")
# plt.show()


# 1.全局阈值法
ret, mask_all = cv2.threshold(src=img,                  # 要二值化的图片
                              thresh=127,               # 全局阈值
                              maxval=255,               # 大于全局阈值后设定的值
                              type=cv2.THRESH_BINARY)   # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
print("全局阈值的shape: ", mask_all.shape)
plt.subplot(2, 2, 2)
plt.imshow(mask_all, cmap='gray')   #cmap = 'gray'是以灰度图形式展示的意思
plt.title("全局阈值")
# plt.show()



# # 2.自适应阈值法
mask_local = cv2.adaptiveThreshold(src=img,                                     # 要进行处理的图片
                                   maxValue=255,                                # 大于阈值后设定的值
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,   # 自适应方法，ADAPTIVE_THRESH_MEAN_C：表区域内均值；ADAPTIVE_THRESH_GAUSSIAN_C：表区域内像素点加权求和
                                   thresholdType=cv2.THRESH_BINARY,             # 超过阈值的值为最大值，其他值是0
                                   blockSize=11,                                # 方阵（区域）大小，
                                   C=1)                                         # 常数项，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
print("局部阈值的shape: ", mask_local.shape)
plt.subplot(2, 2, 3)
plt.imshow(mask_local, cmap='gray')  #cmap = 'gray'是以灰度图形式展示的意思
plt.title("局部阈值")

# 3.OTSU（二值化）确定将图像分成黑白两个部分的阈值
# 图像中大于0的像素，值替换为maxval=255
ret2, mask_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("OTSU的shape: ", mask_OTSU.shape)
plt.subplot(2, 2, 4)
plt.imshow(mask_OTSU, cmap='gray')
plt.title("二值化")
#
plt.show()

# plt.imshow()函数中的cmap = 'gray'是以灰度图形式展示的意思

# imshow函数详解
# 对于imshow函数，opencv的官方注释指出：根据图像的深度，imshow函数会自动对其显示
# 灰度值进行缩放，规则如下：
#1：如果图像数据类型是8U（8位无符号），则直接显示。
#2：:如果图像数据类型是16U（16位无符号）或32S（32位有符号整数），则imshow函数内部
# 会自动将每个像素值除以256并显示，即将原图像素值的范围由[0~255*256]映射到[0~255]
#3：如果图像数据类型是32F（32位浮点数）或64F（64位浮点数），则imshow函数内部会自动
# 将每个像素值乘以255并显示，即将原图像素值的范围由[0~1]映射到[0~255]（注意：原图
# 像素值必须要归一化



# THRESH_BINARY	超过阈值的值为最大值，其他值是0
# THRESH_BINARY_INV	超过阈值的值为0，其他值为最大值
# THRESH_TRUNC	超过阈值的值等于阈值，其他值不变
# THRESH_TOZERO	超过阈值的值不变，其他值为0
# THRESH_TOZERO_INV	超过阈值的值为0，其他值不变
