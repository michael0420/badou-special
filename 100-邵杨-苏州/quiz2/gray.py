import cv2 as cv
import numpy as np

img = cv.imread('1.jpg')
#cv.imshow('1', img)

gray1 = np.zeros([img.shape[0], img.shape[1]], dtype='uint8')
gray2 = np.zeros([img.shape[0], img.shape[1]], dtype='uint8')


#调用接口
def gray():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

#手动转灰度
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray1[i, j] = int(img[i, j, 0]*0.11 + img[i, j, 1]*0.59 + img[i, j, 2]*0.3)

#均值二值化
middle = np.mean(gray1)
for i in range(gray2.shape[0]):
    for j in range(gray2.shape[1]):
        if gray1[i, j] <= middle:
            gray2[i, j] = 0
        else:
            gray2[i, j] = 255

cv.imshow('gary1', gray1)
cv.imshow('gary2', gray2)
cv.imshow('gary', gray())
cv.waitKey(0)
