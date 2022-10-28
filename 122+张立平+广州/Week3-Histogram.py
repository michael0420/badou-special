import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
变为直方图：计算每个灰度图像素值的数量

calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围

cv2.imread()有两个参数，第一个参数filename是图片路径，第二个参数flag表示图片读取模式，共有三种：
彩色模式1；灰度模式0-会产生偏差，故灰度图均采用原图+BGR2GRAY；
包括alpha通道-1；
'''

imgGray1=cv2.imread("lenna.png",0)
img=cv2.imread("lenna.png",1)           #=Oimg=cv2.imread("lenna.png")
imgGray2=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("imgGray1\n",imgGray1)
print("imgGray2\n",imgGray2)

#方法一，用matplotlib的hist()
plt.figure()
plt.hist(imgGray2.ravel(),256)      #将多为数组降为一维
plt.show()


#方法二，用opencv的calcHist()
imgcvHist = cv2.calcHist([imgGray2],[0],None,[256],[0,256])
#设置画布
plt.figure()
plt.title("GrayHist")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(imgcvHist)
plt.xlim([0,256])
plt.show()


#彩色直方图处理
chans = cv2.split(img)      #分为B,G,R通道
colors = ("B","G","R")
plt.figure()
plt.title("Flattened Color Hist")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])
plt.show()




'''
equalizeHist—直方图均衡化
函数原型：equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

#灰度图均衡化
dstGray = cv2.equalizeHist(imgGray2)
#显示均衡化后的直方图
dstGrayHist = cv2.calcHist([dstGray],[0],None,[256],[0,256])
plt.figure()
plt.title("dstGrayHist")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(dstGrayHist)
plt.xlim([0,256])
plt.show()

plt.figure()
plt.hist(dstGray.ravel(),256)
plt.show()

#对比灰度图均衡化前后
cv2.imshow("Before and after",np.hstack([imgGray2,dstGray]))
cv2.waitKey()



#彩色图均衡化， 需要对每个通道均衡化
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
#合并通道
dstColor=cv2.merge((bH,gH,rH))

plt.figure()
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
chans = cv2.split(dstColor)
colors = ("B","G","R")
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()

cv2.imshow("dstColor", dstColor)
cv2.waitKey()