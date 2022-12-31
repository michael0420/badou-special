import cv2
import numpy as np


image = cv2.imread('./data/lenna.png')
median_blur_image = cv2.medianBlur(np.uint8(image), 3)
cv2.imshow('image', image)
cv2.imshow('median_blur_image', median_blur_image)

cv2.waitKey(0)
