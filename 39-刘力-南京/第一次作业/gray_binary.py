from PIL import Image
import numpy as np
import cv2

# 灰度化

# 自己实现
img = Image.open('lenna.png').convert('RGB')  # 打开图片
img = np.array(img)  # 转numpy数组
new = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11  # 转灰度
new = new.astype(int) # 将数组中的float转为int
gray = Image.fromarray(new) # numpy转PIL图像
gray.show() # 显示图像

# 调用接口
pic = cv2.imread('lenna.png', 0)
cv2.imshow('gray', pic)
cv2.waitKey()

# 二值化
binary = np.where(new >= 128, 255, 0)
binary = Image.fromarray(binary)
binary.show()
