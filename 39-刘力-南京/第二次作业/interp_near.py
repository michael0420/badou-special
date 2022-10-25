import numpy as np
import cv2


def interp(img):
    h, w, c = img.shape
    empty = np.zeros((800, 800, c), np.uint8)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            empty[i, j] = img[x, y]

    return empty

img = cv2.imread('lenna.png')
interp_img = interp(img)
cv2.imshow('origin', img)
cv2.imshow('bigger', interp_img)
print(interp_img.shape)
cv2.waitKey()