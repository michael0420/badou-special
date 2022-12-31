"""
@author: Rai
K-Means聚类:彩色图像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
retval, bestLabels, centers =cv2.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
输入：
    data：聚类数据，最好是np.flloat32类型的N维点集，每个特征放一列
    K：聚类类簇数
    bestLabels：输出的整数数组，用于存储每个样本的聚类标签索引
    criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts：重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags：初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS，cv2.KMEANS_RANDOM_CENTERS
          --cv2.KMEANS_PP_CENTERS：KMeans++算法，在聚类中心的初始化过程中的基本原则是使得初始的聚类中心之间的
                                   相互距离尽可能远
          --cv2.KMEANS_RANDOM_CENTERS：随机选择
    centers：集群中心的输出矩阵，每个集群中心为一行数据

输出：
    compactness：紧密度，返回每个点到相应重心的距离的平方和
    labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...等
    centers：由聚类的中心组成的数组
"""

img = cv2.imread('./lenna.png')
h, w, _ = img.shape

# 图像数据转化成一行
data = img.reshape((h * w, 3))
data = np.float32(data)

# 聚类类簇数
# k_array = [2, 4, 8, 16, 32, 64, 128]
k_array = [2, 4, 8, 16, 32]

# 迭代模式选择
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

# 初始中心选择
flags = cv2.KMEANS_PP_CENTERS

# KMeans迭代次数
attempts = 10

compactness_a, labels, centers = [], [], []
for k in k_array:
    compactness, label, center = cv2.kmeans(data, k, None, criteria, attempts, flags)
    compactness_a.append(compactness)
    labels.append(label)
    centers.append(center)

imgs_dst = []
for i, (center, label) in enumerate(zip(centers, labels)):
    center = np.uint8(center)
    img_dst = center[label.flatten()]
    img_dst = img_dst.reshape((h, w, 3))
    imgs_dst.append(img_dst)

titles = ['Original Image', 'KMeans ']
images = [img]
images.extend(imgs_dst)

plt.figure(figsize=(40, 25))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if i == 0:
        plt.title(titles[0], fontsize=60, color='blue')
    else:
        plt.title(titles[1] + 'k=' + str(k_array[i-1]) + '', fontsize=60, color='blue')
    plt.axis('off')
plt.savefig('KMeans_RGB Image.png')
plt.show()
