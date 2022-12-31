import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

image = cv2.imread('../data/lenna.png')
# 将三维转为2维
data = np.float32(image.reshape((-1, 3)))

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
_, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
_, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
_, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
_, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape(image.shape)

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape(image.shape)

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(image.shape)

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape(image.shape)

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape(image.shape)

# 转回RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
image4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
image8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
image16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
image64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4', '聚类图像 K=8', '聚类图像 K=16', '聚类图像 K=64']
images = [image, image2, image4, image8, image16, image64]

for index in range(len(images)):
    plt.subplot(len(images) // 3, 3, index + 1)
    plt.imshow(images[index], 'gray')
    plt.title(titles[index])
plt.show()
