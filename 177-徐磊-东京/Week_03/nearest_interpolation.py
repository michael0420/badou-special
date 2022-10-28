import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

'''
nearest_interpolation
'''


def nearest_interpolation(img_src, height, width):
    h_src, w_src, ch = img_src.shape
    img_dst = np.zeros([height, width, ch], dtype=img_src.dtype)
    ratio_h = h_src / height
    ratio_w = w_src / width
    for i in range(height):
        for j in range(width):
            i_src = int(math.floor(i * ratio_h))
            j_src = int(math.floor(j * ratio_w))
            img_dst[i, j] = img_src[i_src, j_src]

    return img_dst


img_src = cv2.imread("../lenna.png")
img_dst = nearest_interpolation(img_src, 1024, 1024)

cv2.imshow("Original image", img_src)
print("The shape of original image:", img_src.shape)

print("The shape of upsampling image:", img_dst.shape)
cv2.imshow("Upsampling image by nearest_interp", img_dst)
cv2.imwrite("Upsampling by nearest interpolation.png", img_dst)
cv2.waitKey()

