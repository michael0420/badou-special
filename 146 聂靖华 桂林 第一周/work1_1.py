import cv2
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
# 灰度图
img = cv2.imread('lenna.png')
h,w = img.shape[:2]
temp = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        temp[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #BGR转浮点的公式，记住取整数，因为像素在0-255整数
img_gray = temp
print('---img_gray---')
print(img_gray)

plt.subplot(231)
img_ori = plt.imread('lenna.png')
plt.imshow(img_ori)

plt.subplot(232)
plt.imshow(img_gray,cmap='gray')

#灰度图，命令
img2 = cv2.imread('lenna.png')
img2 = rgb2gray(img2)
img_gray2 = img2
plt.subplot(233)
plt.imshow(img_gray2,cmap='gray')

#二值图，手动
img3 = cv2.imread('lenna.png')
img3 = rgb2gray(img3)
h2,w2 = img3.shape[:2]
temp2 = np.zeros([h2,w2],img3.dtype)
for i2 in range(h2):
    for j2 in range(w2):
        # 灰度图转二值图的公式
        if img3[i2,j2]<=0.5:
            img3[i2, j2] = 0
        else:
            img3[i2, j2] = 1
print('---img_binary---')
print(img3)
plt.subplot(234)
plt.imshow(img3,cmap='gray')

img4 = cv2.imread('lenna.png')
img4 = rgb2gray(img4)
img4 = np.where(img4 >= 0.5,1,0)
plt.subplot(235)
plt.imshow(img4,cmap='gray')
plt.show()

