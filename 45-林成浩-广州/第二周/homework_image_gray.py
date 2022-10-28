# -*- coding: utf-8 -*-
# 作业：实现RGB2Gray 及 二值化
# @author：林成浩

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['Hei']  # 指定默认字体
plt.figure(figsize=(14, 14))
plt.rcParams['font.size'] = 20

# 原图
plt.subplot(231)
img = plt.imread("lenna.png")
plt.imshow(img)
plt.title("原图")
print("---image lenna----")
print(img)

# 灰度化
i_lenna = cv2.imread("lenna.png")
print(i_lenna)
h, w = i_lenna.shape[:2]
new_lenna = np.zeros([h, w], i_lenna.dtype)
for i in range(h):
    for j in range(w):
        m = i_lenna[i, j]
        new_lenna[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print(new_lenna)
print("image shou gray: %s" % new_lenna)
# cv2.imshow("image show gary", new_lenna)
# cv2.waitKey(0)
plt.subplot(232)
plt.imshow(new_lenna)
plt.title("self rgb2gray")

# 灰度化 cv2 rgb2gray
i_gary = cv2.cvtColor(i_lenna, cv2.COLOR_BGR2GRAY)
plt.subplot(233)
plt.imshow(i_gary)
plt.title("cv2 rgb2gray")

# 灰度化
i_gary = rgb2gray(i_lenna)
plt.subplot(234)
plt.imshow(i_gary, cmap='gray')
plt.title("plt rgb2gray")

# 二值化
rows, cols = i_gary.shape
for i in range(rows):
    for j in range(cols):
        if i_gary[i, j] <= 0.5:
            i_gary[i, j] = 0
        else:
            i_gary[i, j] = 1

plt.subplot(235)
plt.imshow(i_gary)
plt.title("self binary")

# 二值化
i_gary = np.where(i_gary >= 0.5, 1, 0)
plt.subplot(236)
plt.imshow(i_gary, cmap='gray')
plt.title("np binary")
plt.show()


