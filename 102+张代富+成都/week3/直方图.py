"""
灰度图和彩色图的直方图均衡化是不同的，彩色图的直方图均衡化需要对RGB分别进行均衡化的处理
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
image = cv2.imread("../data/lenna.png", 0)
print(image.shape)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(image)

# 彩色图像直方图均衡化
image2 = cv2.imread("../data/lenna.png", 1)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
b, g, r = cv2.split(image2)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
dst2 = cv2.merge((bH, gH, rH))

# 获取直方图
hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([dst], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([image2], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([dst2], [0], None, [256], [0, 256])

plt.subplot(621)
plt.hist(image.ravel(), range(0, 256))
plt.title("image")
plt.subplot(622)
plt.title("image-hist")
plt.plot(hist1)
plt.subplot(623)
plt.title("hist")
plt.hist(dst.ravel(), range(0, 256))
plt.subplot(624)
plt.title("dst-hist")
plt.plot(hist2)
plt.subplot(625)
plt.title("rgb-hist")
plt.plot(range(0, 256), hist3)
plt.subplot(626)
plt.title("dst-rgb-hist")
plt.plot(range(0, 256), hist4)
plt.show()

cv2.imshow("equalizeHist", np.hstack([image, dst]))
cv2.imshow("equalizeHist2", np.hstack([image2, dst2]))
cv2.waitKey(0)
