import numpy as np
import cv2 as cv
import random

def jiaoyan(src, per):
    noiseimg = src
    noisenum = int(src.shape[0]*src.shape[1]*per)
    for i in range(noisenum):
        y = random.randint(0, src.shape[0] - 1) #随机生成行
        x = random.randint(0, src.shape[1] - 1) #随机生成列
        if random.random() < 0.5:
            noiseimg[y, x] = 0
        else:
            noiseimg[y, x] = 255

    return noiseimg

if __name__ == '__main__':
    img = cv.imread("lenna.png", 0)
    img2 = jiaoyan(img, 0.2)
    img = cv.imread('lenna.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("jiaoyan and gray", np.hstack([img2, gray]))
    #cv.imshow("image jiaoyan", img2)
    cv.waitKey(0)
