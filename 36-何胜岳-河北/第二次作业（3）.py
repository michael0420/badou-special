#========================= 3实现直方图的均衡化============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
calcHist-计算图像直方图
函数原型：calcHist(images,channels,mask,histSize,ranges,hist=None,accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

#将无损原图转化为灰度图
img=cv2.imread('pl.jpg',2|4)               #读取无损原图
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #图像灰度化处理gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# cv2.imshow("image",gray)                      #显示图像
# cv2.waitKey(0)

#equalizeHist—直方图均衡化
dst = cv2.equalizeHist(gray)

#calcHist-计算图像直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
# img.ravel()可以将图片转化成一维数组，直方图的信息都是从这里提取出来的

plt.show()

plt.figure()                          #新建一个图像
plt.title("Grayscale Histogram")      #图像的标题
plt.xlabel("X")                       #X轴标签
plt.ylabel("Y")                       #Y轴标签
plt.plot(dst)
plt.xlim([0,256])                     #设置x坐标轴范围
plt.show()                            #显示图像

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)



# ===========================彩色图像直方图均衡化============================
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)
