import numpy as np
import cv2 as cv

ori = cv.imread('lenna.png', 0)
dst = cv.equalizeHist(ori)
cv.imshow("lenna",np.hstack([ori, dst]))
cv.waitKey(0)


