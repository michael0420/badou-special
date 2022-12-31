"""
@author: Rai
随机采样一致性(random sample consensus)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Least_Squares import least_squares


nums = 300  # 样本的数量
X = np.linspace(0, 50, nums)
A, B = 2, 10
Y = A * X + B

# 直线随机噪声
random_x = X + np.random.normal(size=nums)
random_y = Y + np.random.normal(size=nums)

# 任意噪声
x_noise = []
y_noise = []
for i in range(nums):
    x_noise.append(random.uniform(0, 50))
    y_noise.append(random.uniform(10, 110))
x_noise = np.array(x_noise)
y_noise = np.array(y_noise)
random_X = np.hstack([random_x, x_noise])
random_Y = np.hstack([random_y, y_noise])

# 画散点图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('RANSAC')
ax.scatter(random_x, random_y, c='k', label='RANSAC_data')
ax.scatter(x_noise, y_noise, c='y', label='noise data')
ax.set_xlabel('x')
ax.set_ylabel('y')


# 使用RANSACked算法估算模型
iteration = 10000  # 最大迭代次数
diff = 0.01  #  模型与真实数据之间的误差
best_a = 0  #  最佳模型的斜率
best_b = 0  # 最佳模型的截距
max_inlier = 0  # 最大内点数目
p = 0.99  # 希望得到模型的正确概率
for i in range(iteration):
    #  随机在数据中选取两个点求解模型
    sample_id = random.sample(range(nums*2), 2)
    i, j = sample_id
    x1 = random_X[i]
    x2 = random_X[j]
    y1 = random_Y[i]
    y2 = random_Y[j]

    # 由 y = a*x + b -> a, b
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    # 计算内点数目
    total_inlier = 0
    for i in range(nums * 2):
        y_pred = a * random_X[i] + b
        error = abs(y_pred - random_Y[i])
        if error < diff:
            total_inlier += 1

   # 通过内点数目判断当前模型是否比之前模型好
    if total_inlier > max_inlier:
        iteration = math.log(1 - p) / math.log(1 - pow((total_inlier / nums * 2), 2))
        max_inlier = total_inlier
        best_a = a
        best_b = b

    # 判断内点数目是否达到数据的95%
    if total_inlier > int(nums * 0.95):
        break

Y_pred = best_a * random_X + best_b
Y = A * random_X + B
k, b = least_squares(random_X, random_Y)
Y_linear = k * random_X + b
print('best_a:', best_a)
print('best_b:', best_b)
ax.plot(random_X, Y_pred, color='r', label='RANSAC fit')
ax.plot(random_X, Y, color='b', label='Real data')
ax.plot(random_X, Y_linear, color='c', label='Linear fit')
ax.legend(loc='upper left')
plt.savefig('RANSAC.png')
plt.show()