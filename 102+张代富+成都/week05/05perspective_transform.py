import cv2
import numpy as np

image = cv2.imread('./data/photo.jpg')

src = np.float32([[207, 155], [517, 287], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 生成透视矩阵
m = cv2.getPerspectiveTransform(src, dst)
image2 = cv2.warpPerspective(image, m, (337, 488))
cv2.imshow('image', image)
cv2.imshow('image2', image2)

cv2.waitKey()
