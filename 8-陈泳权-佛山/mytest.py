from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("lenna.png")
w, h = img.shape[:2]
img_gray = np.zeros([w, h], img.dtype)
#自己实现灰度化
for i in range(w):
    for j in range(h):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
cv2.imshow("img gray",img_gray)
# cv2.waitKey(0)

plt.subplot(221)
img = plt.imread("lenna.png")
print(img)
plt.imshow(img)

#调用接口实现灰度化
plt.subplot(222)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')

img_2gray = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_2gray, cmap='gray')
plt.show()


