import numpy as np
import cv2 as cv
import random

def gaosi(src,means,sigma, per):
    noiseimg = src
    noisenum = int(src.shape[0]*src.shape[1]*per)
    for i in range(noisenum):
        y = random.randint(0, src.shape[0] - 1)
        x = random.randint(0, src.shape[1] - 1)
        noiseimg[y, x] = noiseimg[y, x] + random.gauss(means, sigma)
        if noiseimg[y, x] < 0:
            noiseimg[y, x] = 0
        elif noiseimg[y, x] > 255:
            noiseimg[y, x] = 255

    return noiseimg

if __name__ == '__main__':
    img = cv.imread("lenna.png", 0)
    img2 = gaosi(img, 2, 4, 0.01)
    img = cv.imread("lenna.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gaosi and gray", np.hstack([img2, gray]))
    cv.waitKey(0)