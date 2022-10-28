import cv2 as cv
import numpy as np
import random

def add_guassian_noise(img):
    dst = img.copy()
    for channel in range(img.shape[2]):
        for width in range(img.shape[1]):
            for height in range(img.shape[0]):
                if np.random.rand() < 0.5:
                    dst[height, width, channel] = int(dst[height, width, channel] + random.gauss(2, 4))
                    if dst[height, width, channel] > 255:
                        dst[height, width, channel] = 255
                    if dst[height, width, channel] < 0:
                        dst[height, width, channel] = 0
    return dst

def add_impulse_noise(img):
    dst = img.copy()
    for channel in range(img.shape[2]):
        for width in range(img.shape[1]):
            for height in range(img.shape[0]):
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        dst[height, width, channel] = 0
                    else:
                        dst[height, width, channel] = 255
    return dst

if __name__ == "__main__":
    img = cv.imread(r'week2\lenna.png')
    dst_guass = add_guassian_noise(img)
    dst_impulse = add_impulse_noise(img)
    cv.imshow("lenna", np.concatenate((img, dst_guass, dst_impulse), 1))
    cv.waitKey(0)