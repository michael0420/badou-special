"""
@author:Rai
canny 检测边缘
"""

import cv2
import numpy as np

img = cv2.imread('lenna.png')
img_gray = np.uint8(img.mean(axis=-1))

img_cany = cv2.Canny(img_gray, 18, 54, L2gradient=True)
cv2.imshow('Canny img', img_cany)
cv2.imwrite('Canny img_auto.png', img_cany)
cv2.waitKey(0)