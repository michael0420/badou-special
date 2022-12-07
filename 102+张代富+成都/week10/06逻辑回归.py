import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def initialize_with_zeros(shape):
    """
    创建一个形状为 (shape, 1) 的w参数和b=0.
    return:w, b
    """
    w = np.random.normal(loc=1, scale=1.0, size=(shape, 1))
    # w = np.zeros((shape, 1))
    b = random.random()
    return w, b


def basic_sigmoid(x):
    """
    计算sigmoid函数
    """
    return 1 / (1 + np.exp(-x))


def propagate(w, b, x, y):
    """
    参数：w,b,X,Y：网络参数和数据
    Return:
    损失cost、参数W的梯度dw、参数b的梯度db
    """
    m = x.shape[1]

    # w (n,1), x (n, m)
    a = basic_sigmoid(np.dot(w.T, x) + b)
    # 计算损失
    cost = -1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    dz = a - y
    dw = 1 / m * np.dot(x, dz.T)
    db = 1 / m * np.sum(dz)

    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, x, y, num_iterations, learning_rate):
    """
    参数：
    w:权重,b:偏置,X特征,Y目标值,num_iterations总迭代次数,learning_rate学习率
    Returns:
    params:更新后的参数字典
    grads:梯度
    costs:损失结果
    """

    costs = []
    dw, db = None, None
    for i in range(num_iterations):

        # 梯度更新计算函数
        grads, cost = propagate(w, b, x, y)

        # 取出两个部分参数的梯度
        dw = grads['dw']
        db = grads['db']

        # 按照梯度下降公式去计算
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if i % 10 == 0:
            print(f"第{i}次迭代,损失结果:{cost}, b:{b}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def predict(w, b, x):
    """
    利用训练好的参数预测
    return：预测结果
    """

    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)

    # 计算结果
    a = basic_sigmoid(np.dot(w.T, x) + b)

    for i in range(a.shape[1]):
        if a[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1

    return y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.0001):
    """
    """
    # 修改数据形状
    x_train = x_train.reshape(-1, x_train.shape[0])
    x_test = x_test.reshape(-1, x_test.shape[0])
    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])
    # 1、初始化参数
    w, b = initialize_with_zeros(x_train.shape[0])

    # 2、梯度下降
    # params:更新后的网络参数
    # grads:最后一次梯度
    # costs:每次更新的损失列表
    params, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate)

    # 获取训练的参数
    # 预测结果
    w = params['w']
    b = params['b']
    y_prediction_train = predict(w, b, x_train)
    y_prediction_test = predict(w, b, x_test)

    # 打印准确率
    print("训练集准确率: {} ".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("测试集准确率: {} ".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


if __name__ == '__main__':
    data_x, data_y = make_classification(n_samples=500, n_features=50, n_classes=2, n_informative=30, random_state=7)
    x_train_, x_test_, y_train_, y_test_ = train_test_split(data_x, data_y, test_size=0.3, random_state=66)
    model(x_train_, y_train_, x_test_, y_test_, num_iterations=2000, learning_rate=0.0001)
