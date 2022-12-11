import cv2
import numpy as np

def function(img):
    H ,W ,C = img.shape
    KOimg = np.zeros((800,800,C),np.uint8)#生成一个三通道800*800的黑色空白图片
    SH = 800/H # 长度放大倍数
    SW = 800/W # 宽度放大倍数

    for i in range(800):
        for j in range(800):
            X = int(i/SW + 0.5)
            Y = int(j/SH + 0.5)
            KOimg[i,j] = img[X,Y]
    return KOimg

img1 = cv2.imread('lenna.png')
cv2.imshow('orginal',img1)
img2 = function(img1)
cv2.imshow('result',img2)
cv2.waitKey()

