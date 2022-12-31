from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2


# 第一列表示球员每分钟助攻数：assists_per_minute
# 第二列表示球员每分钟得分数：points_per_minute
def player():
     X = [[0.0888, 0.5885],
          [0.1399, 0.8291],
          [0.0747, 0.4974],
          [0.0983, 0.5772],
          [0.1276, 0.5703],
          [0.1671, 0.5835],
          [0.1306, 0.5276],
          [0.1061, 0.5523],
          [0.2446, 0.4007],
          [0.1670, 0.4770],
          [0.2485, 0.4313],
          [0.1227, 0.4909],
          [0.1240, 0.5668],
          [0.1461, 0.5113],
          [0.2315, 0.3788],
          [0.0494, 0.5590],
          [0.1107, 0.4799],
          [0.1121, 0.5735],
          [0.1007, 0.6318],
          [0.2567, 0.4326],
          [0.1956, 0.4280]
          ]

     # print(X)
     clf = KMeans(n_clusters=3)
     y_pred = clf.fit_predict(X)
     print("y_pred:%s" % y_pred)

     # 画图
     x = [n[0] for n in X]
     y = [n[1] for n in X]

     plt.scatter(x, y, c=y_pred, marker='x')

     # 绘制标题
     plt.title("Kmeans-Basketball Data")
     # 绘制x轴和y轴坐标
     plt.xlabel("assists_per_minute")
     plt.ylabel("points_per_minute")
     plt.legend(['A', 'B', 'C'])


def img_kmeans():
     img = cv2.imread(r"D:\JetBrainsProjects\PycharmProjects\CV\lenna.png", 0)
     rows, cols = img.shape

     # 图像二维转一维
     data = img.reshape((rows * cols, 1))
     data = np.float32(data)
     # 停止条件 (type,max_iter,epsilon)
     criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.1)
     flags = cv2.KMEANS_RANDOM_CENTERS

     # kmeans聚类
     compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

     dst = labels.reshape((img.shape[0], img.shape[1]))
     # 运行配置参数中的字体（font）为黑体（SimHei）
     plt.rcParams['font.sans-serif'] = ['SimHei']

     titles = [u'原始图像', u'聚类图像']
     images = [img, dst]
     for i in range(2):
          plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
          plt.title(titles[i])
          # 不显示坐标值
          plt.xticks([]), plt.yticks([])


if __name__ == '__main__':
     player()
     img_kmeans()
     plt.show()
