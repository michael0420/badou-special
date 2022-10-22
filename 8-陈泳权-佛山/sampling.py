import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np

#最接近插值

def Nearest_insert(img, w, h, w1, h1):
    sw = w1/w
    sh = h1/h
    img_change = np.zeros([w1, h1], img.dtype)
    for i in range(w1):
        for j in range(h1):
            x = int(i/sw+0.5)
            y = int(j/sh+0.5)
            img_change[i, j] = img[x, y]
    return img_change

img = cv2.imread("lenna.png")
w, h = img.shape[:2]
img_gray = rgb2gray(img)
print(img_gray)
plt.subplot(221)
plt.imshow(img)
plt.show()
cv2.imshow("img", img_gray)
img_big = Nearest_insert(img_gray, w, h, 800, 800)
print(img_big)
cv2.imshow("img_big", img_big)


# #双线插值
def both_insert(img,w,h,w1,h1):
    sw = float(w)/w1
    sh = float(h)/h1
    img_change = np.zeros([w1, h1], img.dtype)
    for j in range(h1):
        for i in range(w1):
            x = (i+0.5)*sw-0.5
            y = (j+0.5)*sh-0.5
            x1 = int(np.floor(x))
            x2 = min(x1+1,w-1)
            y1 = int(np.floor(y))
            y2 = min(y1+1,h-1)
            temp1 = (x2-x)*img[y1,x1]+(x-x1)*img[y1,x2]
            temp2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]
            img_change[j, i] = float((y2-y)*temp1 + (y-y1)*temp2)
    return img_change
img_small = both_insert(img_gray,w,h,800,800)
print(img_small)
cv2.imshow("both_insert", img_small)
cv2.waitKey(0)