import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

pic_path = r"D:\JetBrainsProjects\PycharmProjects\CV\lenna.png"


# canny实现
def canny():
    image = cv2.imread(pic_path, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # 灰度化
    cv2.imshow("canny", cv2.Canny(gray, 100, 100))  # 实现Canny
    cv2.waitKey()


canny()

img = plt.imread(pic_path)
plt.subplot(221)
plt.imshow(img)

if pic_path[-4:] == ".png":
    img = img * 255
img_gray = img.mean(axis=-1)
plt.subplot(222)
plt.imshow(img_gray, cmap="gray")


def gaussian_smooth():
    sigma = 0.5
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()

plt.show()
