# ===============================K-Means==================================
'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''
import cv2
import  numpy as np
import  matplotlib.pyplot as plt

img=cv2.imread('lenna.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img.shape)

w,h=img.shape[:]

# 图像二维像素转换为一维
data=img.reshape((w*h),1)
data=np.float32(data)
print(data)

#停止条件 (type,max_iter,epsilon)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0) #10为迭代次数，1为精确度
flags=cv2.KMEANS_RANDOM_CENTERS

cog,labels,centers=cv2.kmeans(data,10,None,criteria,10,flags)

dst=labels.reshape(img.shape[0],img.shape[1])

#正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

titles=['原图','新图']
imgs=[img,dst]

for i in range(2):
   plt.subplot(1,2,i+1)
   plt.imshow(imgs[i],'gray')
   plt.title(titles[i])
plt.show()



#================================K——Means——RGB================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png')
print(img.shape)

#图像二维像素转换为一维
data=img.reshape((-1,3))
data=np.float32(data)
print(data)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags=cv2.KMEANS_RANDOM_CENTERS

compactness,labels2,centers2=cv2.kmeans(data,2,None,criteria,10,flags)
compactness,labels4,centers4=cv2.kmeans(data,4,None,criteria,10,flags)
compactness,labels8,centers8=cv2.kmeans(data,8,None,criteria,10,flags)
compactness,labels16,centers16=cv2.kmeans(data,16,None,criteria,10,flags)
compactness,labels32,centers32=cv2.kmeans(data,32,None,criteria,10,flags)

#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))
#flatten()函数用法
# flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
# flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用！
# a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 。
# a.flatten().A：a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers32 = np.uint8(centers32)
res = centers32[labels32.flatten()]
dst32 = res.reshape((img.shape))

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst2=cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4=cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8=cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16=cv2.cvtColor(dst16,cv2.COLOR_BGR2RGB)
dst32=cv2.cvtColor(dst32,cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

titles=['k=0','k=2','k=4','k=8','k=16','k=32']
imgs=[img,dst2,dst4,dst8,dst16,dst32]

for i in range(6):
   plt.subplot(2,3,i+1)
   plt.imshow(imgs[i],'gray')
   plt.title(titles[i])
plt.show()


#=======================K-Means_athlete======================================
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1：导入数据
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

# 输出数据集
print(X)

# KMeans聚类
clf = KMeans(n_clusters=3)  # 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_ = clf.fit_predict(X)  # 载入数据集X，并且将聚类的结果赋值给y_

print(clf)
print('y_', y_)

# 3：输出图像
# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
y = [n[1] for n in X]
print(x)
print(y)

plt.scatter(x, y, s=300, c=y_, marker='*')

plt.title('篮球运动员比赛数据')
plt.xlabel('每分钟助攻数')
plt.ylabel('每分钟得分数')

# plt.legend(["A","B","C"])
plt.legend('A', loc='best')

plt.show()