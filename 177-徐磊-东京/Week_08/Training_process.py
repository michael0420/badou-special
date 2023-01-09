"""
@author:Rai
手推深度学习训练过程
"""

import numpy as np


def train(x, Y, w, b, lr):
    # 1。正向传播
    # (1)输入层 -> 隐藏层
    z11 = x[0] * w[0][0] + x[1] * w[0][1] + 1 * b[0]
    a11 = 1 / (1 + np.exp(-z11))
    z12 = x[0] * w[0][2] + x[1] * w[0][3] + 1 * b[0]
    a12 = 1 / (1 + np.exp(-z12))

    # (2)隐藏层 -> 输出层
    z21 = a11 * w[1][0] + a12 * w[1][1] + 1 * b[1]
    a21 = 1 / (1 + np.exp(-z21))
    z22 = a11 * w[1][2] + a12 * w[1][3] + 1 * b[1]
    a22 = 1 / (1 + np.exp(-z22))

    y_pred = np.array([a21,a22])

    # 2。反向传播
    # (1) 计算误差
    E = np.square(Y - y_pred) / 2
    E = E[0] + E[1]
    # (2) 隐藏层h1 -> 输出层梯度计算
    E_a21_gradient = a21 - Y[0]
    a21_z21_gradient = a21 * (1 - a21)
    z21_w21_gradient = a11
    E_w21_gradient = E_a21_gradient * a21_z21_gradient * z21_w21_gradient
    E_w22_gradient = (a21 - Y[0]) * a21_z21_gradient * a12

    a22_z22_gradient = a22 * (1 - a22)
    E_w23_gradient = (a22 - Y[1]) * a22_z22_gradient * a11
    E_w24_gradient = (a22 - Y[1]) * a22_z22_gradient * a12

    # (3) 输入层 -> 隐藏层梯度计算
    E1_a11_gradient = (a21 - Y[0]) * a21 * (1.0 - a21) * w[1][0]
    E2_a11_gradient = (a22 - Y[1]) * a22 * (1.0 - a22) * w[1][2]
    E_a11_gradient = E1_a11_gradient + E2_a11_gradient
    a11_z11_gradient = a11 * (1 - a11)
    z11_w11_gradient = x[0]
    E_w11_gradient = E_a11_gradient * a11_z11_gradient * z11_w11_gradient

    z11_w12_gradient = x[1]
    E_w12_gradient = E_a11_gradient * a11_z11_gradient * z11_w12_gradient

    E1_a12_gradient = (a21 - Y[0]) * a21 * (1 - a21) * w[1][1]
    E2_a12_gradient = (a22 - Y[1]) * a22 * (1 - a22) * w[1][3]
    E_a12_gradient = E1_a12_gradient + E2_a12_gradient
    a12_z12_gradient = a12 * (1 - a12)
    z12_w13_gradient = x[0]
    E_w13_gradient = E_a12_gradient * a12_z12_gradient * z12_w13_gradient

    z12_w14_gradient = x[1]
    E_w14_gradient = E_a12_gradient * a12_z12_gradient * z12_w14_gradient

    # (4) 权值更新
    w_gradient = np.array([[E_w11_gradient, E_w12_gradient, E_w13_gradient, E_w14_gradient],
                           [E_w21_gradient, E_w22_gradient, E_w23_gradient, E_w24_gradient]])

    w = w - lr * w_gradient
    return y_pred, E, w

if __name__ == "__main__":
    x = np.array([0.05, 0.1])
    Y = np.array([0.01, 0.99])
    w = np.array([[0.15, 0.2, 0.25, 0.3],
                  [0.4, 0.45, 0.5, 0.55]])
    b = np.array(([0.35, 0.6]))
    epochs = 10000
    lr = 0.5
    for epoch in range(epochs+1):
        y_pred, E, w = train(x, Y, w, b, lr)
        if epoch % 10 == 0:
            print('epoch:{}, error:{}, 预测:[{},{}]'.format(epoch, E, y_pred[0], y_pred[1]))