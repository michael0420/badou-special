import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# tar = cv2.equalizeHist(img)
#
# plt.figure()
# plt.hist(tar.ravel(), 256)
# plt.show()
#
# cv2.imshow('hist_equalize', np.hstack([img, tar]))
# cv2.waitKey()

img = cv2.imread('lenna.png')
(b, g, r) = cv2.split(img)

b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
tar = cv2.merge((b, g, r))

cv2.imshow('hist_equalize', np.hstack([img, tar]))
cv2.waitKey()