import numpy as np
import matplotlib.pyplot as plt
import random


def least_squares(x, y, n):
    s1, s2, s3, s4 = 0, 0, 0, 0
    for i in range(n):
        s1 = s1 + x[i] * y[i]
        s2 = s2 + x[i]
        s3 = s3 + y[i]
        s4 = s4 + x[i] * x[i]
    k = (n * s1 - s2 * s3) / (s4 * n - s2 * s2)
    b = (s3 - k * s2) / n
    return k, b

def data(size, k, b):   #生成线性数据集
    X = np.linspace(0, 50, size)
    Y = k * X + b
    random_x = list(X)
    random_y = list(Y)
    # 添加直线随机噪声
    for i in range(int(size/2)):
        random_x.append(X[i] + random.uniform(0.5, 0.5))
        random_y.append(Y[i] + random.uniform(-1.5, 1.5))
        # 添加随机噪声
    for i in range(int(size/2)):
        random_x.append(random.uniform(0, 55))
        random_y.append(random.uniform(20, 120))
    RANDOM_X = np.array(random_x)  # 散点图的横轴。
    RANDOM_Y = np.array(random_y)
    return RANDOM_X, RANDOM_Y

'''
n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
iterations = 0
best_model (k, b) = 0， 0
best_total_X, Y = []
best_error = 无穷大
while(iterations < k)
  maybe_inliers = 从数据集中随机选择n个点
  maybe_model（k, b） = 适合于maybe_inliers的模型参数
  maybe_totalX,Y = maybe_inliers

  for(每个数据集中不属于maybe_inliers的点)
    if(如果点适合于maybe_model，且错误小于t)
      将点添加到maybe_total
  if(maybe_total中的元素数目大于d)
    已经找到好的模型，现在测试该模型到底有多好
    better_model = 适合于maybe_total中所有点的模型参数
    this_error = better_model究竟如何适合这些点的度量
    if(this_error < best_error)
      我们发现了比以前更好的模型，保存该模型直到更好的模型出现
      best_model = better_model
      best_error = this_error
  iteration++

return best_model, best_error
'''

def ransac(X, Y, n, K, t, d):
    iterations = 0
    best_k, best_b = 0, 0
    best_error = np.inf
    while iterations < K:
        maybe_inliers_X = []
        maybe_inliers_Y = []
        sample_index = random.sample(range(len(X)), n)
        for i in sample_index:
            maybe_inliers_X.append(X[i])
            maybe_inliers_Y.append(Y[i])
        maybe_total_X, maybe_total_Y = maybe_inliers_X, maybe_inliers_Y
        maybe_outers_X = list(set(X) - set(maybe_inliers_X))
        maybe_outers_Y = list(set(Y) - set(maybe_inliers_Y))
        k, b = least_squares(maybe_inliers_X, maybe_inliers_Y, n)

        for x, y in zip(maybe_outers_X, maybe_outers_Y):
            y_estimate = k * x + b
            if abs(y_estimate - y) < t:
                maybe_total_X.append(x)
                maybe_total_Y.append(y)
        if len(maybe_total_X) > d:
            this_error = []
            k, b = least_squares(maybe_total_X, maybe_total_Y, len(maybe_total_X))
            for x, y in zip(maybe_total_X, maybe_total_Y):
                y_estimate = k * x + b
                this_error.append(abs(y_estimate - y))
            this_error = np.mean(this_error)
            if this_error < best_error:
                best_b = b
                best_k = k
                best_error = this_error

        iterations += 1
    return best_k, best_b, best_error


if __name__ == "__main__":
    data = data(300, 2, 10)  # 300+300个点， y = 2x + 10
    ransac = ransac(data[0], data[1], 50, 500, 1.5, 80)
    print("Coeff: {} Intercept: {} error:{}".format(ransac[0], ransac[1], ransac[2]))

    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(data[0], data[1], 'bo', label='original data')
    c = np.linspace(0, 55, 3)
    plt.plot(c, 2 * c + 10, 'y--', label='liner fit')
    plt.plot(c, ransac[0] * c + ransac[1], 'r', label='ransac fit')
    plt.legend()
    plt.show()



