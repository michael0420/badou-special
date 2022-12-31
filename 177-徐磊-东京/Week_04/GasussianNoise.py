# -*- coding: utf-8 -*-
"""
@author:rai
高斯噪声
"""

import numpy as np
import cv2
import random


def GaussianNoise(img, mean, sigma, ratio):
    noise_img = img.copy()
    pixels_total = img.shape[0] * img.shape[1]
    noise_num = int(np.floor(pixels_total * ratio))
    rand_num = np.array(random.sample(range(0, pixels_total), noise_num))  # 产生不重复的随机数
    x = rand_num // img.shape[1]
    y = rand_num % img.shape[1]
    # noise_img[x, y] = noise_img[x, y] + int(random.gauss(mean, sigma))
    gauss_img = np.random.normal(mean, sigma, img.shape)
    noise_img[x, y] = (noise_img + gauss_img)[x, y]
    noise_img = np.uint8(np.clip(noise_img, 0, 255))

    return noise_img


if __name__ == '__main__':
    color = False
    if color:
        img = cv2.imread('lenna.png')
        img_gaussian = GaussianNoise(img, 1, 4, 0.8)
        img_merge = np.hstack([img, img_gaussian])
        cv2.imwrite('Gaussian Image_Color.png', img_merge)
    else:
        img = cv2.imread('lenna.png', 0)
        img_gaussian = GaussianNoise(img, 1, 4, 0.01)
        img_merge = np.hstack([img, img_gaussian])
        cv2.imwrite('Gaussian Image_Gray.png', img_merge)
    cv2.imshow('Left: Original Image, Right: Gaussian Image', img_merge)

    cv2.waitKey(0)
