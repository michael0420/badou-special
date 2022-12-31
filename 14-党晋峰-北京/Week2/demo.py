# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
"""
彩色图像的灰度化、二值化
"""

# 原图
plt.subplot(221)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.title('img')
plt.imshow(img)
print("---image lenna----")
print(img)


# 灰度化算法
plt.subplot(222)
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(img_gray)
img_gray1 = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
print("image show gray: %s" % img_gray1)
plt.title('image show gray')
plt.imshow(img_gray1, cmap='gray')

# 灰度化函数实现
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.title('cv.img_gray')
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 127, 255, 0)
print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(224)
plt.title('img_binary')
plt.imshow(img_binary, cmap='gray')

plt.show()
