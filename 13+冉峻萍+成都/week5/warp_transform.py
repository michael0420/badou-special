import cv2
import numpy as np

img = cv2.imread("photo1.jpg")

src = np.float32([[207,151], [517,285], [17,601], [343,731]])
dst = np.float32([[0,0], [337,0], [0,488], [337,488]])

# 生成变换矩阵
m = cv2.getPerspectiveTransform(src, dst)

print(m)

# 进行透视变换
result = cv2.warpPerspective(img, m, (337, 488))

cv2.imshow("img", img)

cv2.imshow("result", result)

cv2.waitKey(0)