"""
canny算法open-cv接口
"""

import cv2

image = cv2.imread('./data/lenna.png', 0)
cv2.imshow('canny', cv2.Canny(image, 50, 200))
cv2.waitKey()
cv2.destroyAllWindows()
