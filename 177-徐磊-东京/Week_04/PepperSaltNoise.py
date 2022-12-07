# -*- coding: utf-8 *-*
"""
@author: Rai
椒盐噪声
"""

import random
import numpy as np
import cv2


def PepperSaltNoise(img, ratio):
    noise_img = img.copy()
    pixel_total = img.shape[0] * img.shape[1]
    noise_num = int(pixel_total * ratio)
    rand_point = random.sample(range(0, pixel_total), noise_num)  # 产生不重复的随机数
    for point in rand_point:
        x = point // img.shape[0]
        y = point % img.shape[1]
        noise = random.random()
        noise_img[x, y] = 0 if noise < 0.5 else 255

    return noise_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_salt = PepperSaltNoise(img, 0.05)
    color = True
    if color:
        img_merge = np.hstack([img, img_salt])
        cv2.imwrite('Pepper Salt Image_Color.png', img_merge)
    else:
        img_merge = np.hstack([img[:, :, 0], img_salt[:, :, 0]])
        cv2.imwrite('Pepper Salt Image_Gray.png', img_merge)
    cv2.imshow('Left: Original Gray Image, Right: Spiced Salt Image', img_merge)
    cv2.waitKey(0)
