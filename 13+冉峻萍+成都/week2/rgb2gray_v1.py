import cv2
import numpy as np

img = cv2.imread("Lenna.jpg")

h,w = img.shape[:2]
gray = np.zeros([h,w], img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i,j]
        gray[i,j] = (m[2]*76 + m[1]*151 + m[0]*28) >> 8
        
cv2.imshow("Gray", gray)

cv2.waitKey(0)