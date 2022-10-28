import cv2
import numpy as np
img = cv2.imread('lenna.png', 1)
(b, g, r) = cv2.split(img)  # 通道分解
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH),)  # 通道合成
res = np.hstack((img, result))
cv2.imshow('dst', res)
cv2.waitKey(0)
