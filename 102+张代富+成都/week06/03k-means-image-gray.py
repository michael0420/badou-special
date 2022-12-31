import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 读取原始图像灰度颜色
image = cv2.imread('../data/lenna.png', 0)

# 图像二维像素转换为一维
data = np.float32(image.reshape((image.shape[0] * image.shape[1], 1)))

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像
image_means = labels.reshape((image.shape[0], image.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [image, image_means]
for index in range(len(images)):
    plt.subplot(len(images) // 2, 2, index + 1)
    plt.imshow(images[index], 'gray')
    plt.title(titles[index])
plt.show()
