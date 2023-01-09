"""
@author: Rai
从零实现神经网络
"""

import numpy as np


class Neural_Network(object):
    def __init__(self, input_node, hidden_node, output_node, learn_rate):
        # 定义输入，隐藏层，输出层神经元的个数
        self.in_node = input_node
        self.hid_node = hidden_node
        self.out_node = output_node

        # 定义学习率
        self.lr = learn_rate

        # 输入层 -> 隐藏层权重初始化
        self.weight_in_hid = np.random.normal(loc=0, scale=pow(hidden_node, -0.5), size=(self.hid_node, self.in_node))
        # 隐藏层 -> 输出层权重初始化
        self.weight_hid_out = np.random.normal(loc=0, scale=pow(output_node, -0.5), size=(self.out_node, self.hid_node))

        #隐藏层和输出层的激活函数
        self.activation = lambda x: 1 / (1 + np.exp(-x))

    def train(self, x, Y):
        x = x.reshape(-1, 1)
        # x = np.array(x, ndmin=2).T
        Y = np.array(Y, ndmin=2).T

        # 输入层到隐藏层产生的信号量
        hid_in = np.dot(self.weight_in_hid, x)
        hid_out = self.activation(hid_in)

        #隐藏层到输出层产生的信号量
        final_in = np.dot(self.weight_hid_out, hid_out)
        final_out = self.activation(final_in)

        # 误差
        train_loss = ((final_out - Y) ** 2).sum().mean()
        # 误差对输出层的信号梯度
        loss_final_out_grad = final_out - Y
        # 误差对输出层输入信号的梯度
        loss_final_in_grad = loss_final_out_grad * final_out * (1 - final_out)
        # 误差对隐藏层->输出层之间的权重梯度
        loss_weight_hid_out_grad = np.dot(loss_final_in_grad, hid_out.T)
        self.weight_hid_out -= self.lr * loss_weight_hid_out_grad

        # 误差对隐藏层输出信号的梯度
        loss_hid_out_grad = np.dot(self.weight_hid_out.T, loss_final_in_grad)
        # 隐藏层输出对输入信号的梯度
        hid_out_in_grad = hid_out * (1 - hid_out)
        # 误差对隐藏层输入信号的梯度
        loss_hid_in_grad = loss_hid_out_grad * hid_out_in_grad
        # 误差对输入层->隐藏层之间权重梯度
        loss_weight_in_hid_out = np.dot(loss_hid_in_grad, x.T)
        self.weight_in_hid -= self.lr * loss_weight_in_hid_out
        return train_loss

    def predict(self, x):
        # x = np.array(x, ndmin=2).T
        x = x.reshape(-1, 1)
        # 计算输入->隐藏层的输出
        hid_in = np.dot(self.weight_in_hid, x)
        hid_out = self.activation(hid_in)

        # 计算隐藏层->输出层的输出
        final_in = np.dot(self.weight_hid_out, hid_out)
        final_out = self.activation(final_in)
        return final_out


if __name__ == '__main__':
    train_path = './dataset/mnist_train.csv'
    test_path = './dataset/mnist_test.csv'
    in_nodes = 28 * 28
    hidden_nodes = 300
    out_nodes = 10
    learn_rate = 0.1
    epochs = 10
    net = Neural_Network(input_node=in_nodes,
                         hidden_node=hidden_nodes,
                         output_node=out_nodes,
                         learn_rate=learn_rate)

    # 读取训练数据
    with open(train_path) as f:
        train_data_set = f.readlines()

    # 读取测试数据
    with open(test_path) as f:
        test_data_set = f.readlines()

    for epoch in range(epochs):
        for train_data in train_data_set:
            data = train_data.split(',')
            train_x = data[1:]
            train_x = np.asfarray(train_x)
            # 数据归一化，并使数据大小在【0.01，1】，避免数据过小影响收敛速度和精度
            train_x = train_x / 255.0 * 0.99 + 0.01

            train_y = np.zeros(out_nodes) + 0.01
            label = int(data[0])
            train_y[label] = 0.99
            train_loss = net.train(train_x, train_y)

        # 测试
        acc_count = 0
        for test_data in test_data_set:
            data = test_data.split(',')
            test_x = data[1:]
            test_x = np.array(test_x, dtype='float32')
            test_x = test_x / 255.0 * 0.99 + 0.01

            y_pred = net.predict(test_x)
            idx = y_pred.argmax()
            if idx == int(data[0]):
                acc_count += 1

        test_acc = acc_count / len(test_data_set)
        print('epoch:{}, train_loss:{:.4f}, test_acc:{}'.format(epoch, train_loss, test_acc))