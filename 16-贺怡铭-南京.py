
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lenna.png')
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(imgRGB)

# 灰度图像
h, w = img.shape[0:2]  # 图片的高度和宽度
img_g = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        co = img[i, j]  # 获取待测图片的h和w
        img_g[i, j] = int(co[0] * 0.11 + co[1] * 0.59 + co[2] * 0.3)  # 灰度图像赋值给img_g
print(img_g)
plt.subplot(222)
plt.imshow(img_g, cmap='gray')
# cv2.imshow('image', img_g)
# key=cv2.waitKey()


# 二值化
img_b = np.where(img_g < 99, 0, 1)
plt.subplot(223)
plt.imshow(img_b, cmap='gray')
plt.show()

