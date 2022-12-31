import cv2 as cv
import numpy as np


def nearest(img, h1, w1):
    h0, w0, channel0 = img.shape
    emptyimg = np.zeros((h1, w1, channel0), img.dtype)
    sh = h1 / h0
    sw = w1 / w0
    for i in range(h1):
        for j in range(w1):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyimg[i, j] = img[x, y]
    return emptyimg


img = cv.imread('lenna.png')
dst = nearest(img, 800, 800)
print(dst)
print(dst.shape)
cv.imshow("image",img)
cv.imshow("nearest interp", dst)
cv.waitKey()
