#最邻近插值，放大到1000*1000
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
H,W,channels = img.shape
img_new = np.zeros((1000,1000,channels),np.uint8)
sh = 1000/H #宽放大比例
sw = 1000/W #长放大比例
for i in range(1000):
    for j in range(1000):
        x = int(i/sw + 0.5)
        y = int(j/sh + 0.5)
        img_new[i,j] = img[x,y]


cv2.imshow('ori',img)
cv2.imshow('new',img_new)
cv2.waitKey(0)


