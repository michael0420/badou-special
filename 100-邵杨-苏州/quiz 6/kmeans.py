import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('lenna.png')
print(img.shape)

#图像二维像素转换为1维
data = img.reshape((-1, 3))
data = np.float32(data)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']
k_values = [2, 4, 8, 16, 64]

plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文标签
plt.subplot(2, 3, 1), plt.imshow(img)
plt.title(titles[0])

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0) #停止条件
flags = cv.KMEANS_RANDOM_CENTERS


for i, k in enumerate(k_values):
    compactness, labels, centers = cv.kmeans(data, k, None, criteria, 10, flags) #距离、 原图像上对应点所对应的类别中心的索引、每个类别中心的rgb值
    centers_k = np.uint8(centers)
    res_k = centers_k[labels.flatten()]
    dst_k = res_k.reshape((img.shape))
    dst_k = cv.cvtColor(dst_k, cv.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 2), plt.imshow(dst_k)
    plt.title(titles[i+1])
plt.show()