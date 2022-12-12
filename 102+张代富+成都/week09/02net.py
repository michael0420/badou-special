import numpy
from scipy import special


class NeuralNetWork:
    def __init__(self, ins_node, hd_nodes, out_nodes, lr):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = ins_node
        self.hd_nodes = hd_nodes
        self.out_nodes = out_nodes

        # 设置学习率
        self.lr = lr

        # 初始化权重矩阵  wih:表示输入层和中间层节点间链路权重形成的矩阵  who:表示中间层和输出层间链路权重形成的矩阵
        self.wih = numpy.random.rand(self.hd_nodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.out_nodes, self.hd_nodes) - 0.5

        # self.wih = numpy.random.normal(0.0, pow(self.hd_nodes, -0.5), (self.hd_nodes, self.inodes))
        # self.who = numpy.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hd_nodes))

        self.activation_function = lambda x: special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        """
        根据输入的训练数据更新节点链路权重
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        """
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, ins):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, ins)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def get_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data


def deal_data(data, out_node):
    values = data.split(',')
    ins = (numpy.asfarray(values[1:])) / 255.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系
    labels = numpy.zeros(out_node) + 0.01
    labels[int(values[0])] = 0.99
    return int(values[0]), ins, labels


if __name__ == '__main__':
    # 由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    # 初始化网络
    net = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读入训练数据
    training_data_list = get_data("data/mnist_train.csv")

    # 加入epochs,设定网络的训练循环次数
    epochs = 5
    for e in range(epochs):
        # 把数据依靠','区分，并分别读入
        for record in training_data_list:
            net.train(deal_data(record, output_nodes)[1], deal_data(record, output_nodes)[2])

    # 测试集
    score = 0
    test_data_list = get_data("./data/mnist_test.csv")
    for record in test_data_list:
        correct_number, ins_test, _ = deal_data(record, output_nodes)
        # 预测
        label = numpy.argmax(net.query(ins_test))
        if label == correct_number:
            score += 1
            print(f"正确的数字为:{correct_number}, 预测的结果是:{label}，预测正确！")
        else:
            print(f"正确的数字为:{correct_number}, 预测的结果是:{label}，预测错误！")
    print(f"accuracy={score / len(test_data_list)}")
