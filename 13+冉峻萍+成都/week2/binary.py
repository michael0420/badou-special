import cv2
import numpy as np

img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
        
h,w = img.shape[:2]

print(img.dtype)
for i in range(h):
	for j in range(w):
		if (img[i,j] <= 128):
			img[i,j] = 0
		else :
			img[i,j] = 255

cv2.imshow("binary", img)

cv2.waitKey(0)
