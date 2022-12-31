from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]])
# 层次聚类信息
distance = linkage(x, method='ward')
# 从给定的链接矩阵定义的层次聚类中形成平面聚类
f = fcluster(distance, 4, 'distance')
print(f)
fig = plt.figure(figsize=(4, 3))
dendrogram(distance)
plt.show()
