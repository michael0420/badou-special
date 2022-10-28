import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("Lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0,256])

# 显示直方图
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

# 显示两张对比图片
cv2.imshow("Histogram_equalization", np.hstack([gray, dst]))

cv2.waitKey(0)