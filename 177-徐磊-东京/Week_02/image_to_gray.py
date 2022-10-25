# -*- coding: utf-8 -*-
"""
@author: rai jo

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Gray image (cv)
img_cv = cv2.imread('lenna.png')
h, w = img_cv.shape[:2]

img_gray_cv = np.zeros([h, w], dtype=img_cv.dtype)
for i in range(h):
    for j in range(w):
        dot = img_cv[i][j]
        img_gray_cv[i][j] = int(dot[0] * 0.11 + dot[1] * 0.59 + dot[2] * 0.3)

cv2.imshow('gray image',img_gray_cv)
# cv2.waitKey(0)

# original image
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
img_plt = plt.imread('lenna.png')
# print('img data read by plt', img_plt)
ax1.imshow(img_plt)
plt.title('Original image')

# Gray image by myself
ax2 = fig.add_subplot(2, 2, 2)
img_gray_plt_myself = np.zeros([h, w], dtype=img_plt.dtype)
# for i in range(h):
#     for j in range(w):
#         dot = img_plt[i][j]
#         img_gray_plt_myself[i][j] = dot[0] * 0.3 + dot[1] * 0.59 + dot[2] * 0.11
img_gray_plt_myself = img_plt[:, :, 0] * 0.3 + img_plt[:, :, 1] * 0.59 \
                      + img_plt[:, :, 2] * 0.11
ax2.imshow(img_gray_plt_myself, cmap='gray')
plt.title('Gray image by myself')

# Gray image by api
fig.add_subplot(223)
img_gray_api = rgb2gray(img_plt) # Y = 0.2125 R + 0.7154 G + 0.0721 B
plt.imshow(img_gray_api, cmap='gray')
plt.title('Gray image by api')

# Binary image
fig.add_subplot(224)
img_binary = img_gray_api
# for i in range(h):
#     for j in range(w):
#         if img_binary[i][j] < 0.5:
#             img_binary[i][j] = 0
#         else:
#             img_binary[i][j] = 1
img_binary = np.where(img_binary < 0.5, 0, 1)
plt.imshow(img_binary, cmap='gray')
plt.title('Binary image')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.show()
fig.savefig('Gray and binary image.png')
