# 绘制灰度图像的直方图
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('lenna.png',1)# 1以彩色图读取，0以灰度图读入
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # 方法一：使用hist（）函数，其两个参数：0、数据源，必须为一维数组，使用ravel（）函数可使数据转化为一维数组，
# # 1、像素级一般是256（【0，255】）
# plt.figure()
# plt.hist(img_gray.ravel(),256)
# plt.show()

# 灰度直方图均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
# img = cv2.imread('lenna.png',1)
# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img2 = cv2.equalizeHist(img1) # equlizaHist(src,dst=None):src:单通道图像
# # 直方图
# hist = cv2.calcHist([img2],[0],None,[256],[0,255]) # 统计直方图信息，即每个像素值对应的像素点个数
# # print(hist)
# # print(type(hist))
# # print(hist.shape)
# plt.figure()
# plt.subplot(1,2,1)
# plt.hist(img1.ravel(),256)
# plt.subplot(1,2,2)
# plt.hist(img2.ravel(),256)
# plt.show()
#
# cv2.imshow('Histogram Equalization',np.hstack([img1,img2])) # np.hstack(img1,img2);图像的水平拼接；np.vstack（）垂直拼接
# cv2.waitKey()

# 彩色图像直方图均衡化
import cv2
img = cv2.imread('lenna.png',1)
# 彩色图像需要对每个通道都进行均衡化
b,g,r = cv2.split(img) # split() 对图像进行拆分，返回图像的三个通道
be = cv2.equalizeHist(b)
ge = cv2.equalizeHist(g)
re = cv2.equalizeHist(r)
# 通道合并
img2 = cv2.merge([be,ge,re])
cv2.imshow('result',img2)
cv2.waitKey()
