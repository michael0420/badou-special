import numpy as np


class BPNet:
    """
    手推神经网络
    """
    """
    i1 h1 o1
    i2 h2 02
    激活函数 sigmoid 1/(1+e^-x)
    """

    def __init__(self, rate, iter):
        self.lr = rate  # 学习率
        self.layers = []
        self.iter = iter
        pass

    def add_layer(self, mode):
        """"""
        self.layers.append(mode)
        pass

    def loss(self, res, label):
        loss = np.square(res - label) / 2
        return loss

    def optimizer(self):
        """
        优化器
        :return:
        """
        pass

    def fit(self, data):
        for layer in self.layers:
            data = layer.fit(data)
        return data

    # def get_dp_loss(self,res,label):
    #     #     dp_o=
    #     #     return dp_o
    #     #     pass

    def back_propagation(self, res, label):
        layer_i = len(self.layers)
        dp_o = (res - label)
        while layer_i >= 0:
            layer_i -= 1
            dp_o = self.layers[layer_i].back_propagation(self.lr, dp_o)

    def train_mode(self, train_data):
        """
        训练 模型
        :param train_data:
        :return:
        """
        data = train_data['data']
        label = train_data['label']
        for i in range(self.iter):
            res = self.fit(data)
            loss = self.loss(res, label)
            print("第 {} 次迭代，res:{} \nloss: {}".format(i, res.T, loss.T))
            self.back_propagation(res, label)


class Line:
    def __init__(self, i_n, o_n):
        """
        :param i_n: 输入数量
        :param o_n: 输出数量
        """
        self.i_n = i_n
        self.o_n = o_n
        self.data = None
        self.o = None
        self.dp_w = None
        self.dp_b = None
        self.init_weight()

    def init_weight(self):
        """
        初始化权重
        w: o_n * i_n
        b: o_n *1
        :return:
        """
        self.w = np.random.random((self.o_n,self.i_n))
        self.b = np.random.random((self.o_n,1))

    def sigmoid(self, z):
        """
        激活函数 sigmoid
        :return:
        """
        a = 1 / (1 + np.exp(-z))
        return a

    def fit(self, data):
        self.data = data
        z = np.dot(self.w, data) + self.b
        print(z)
        self.o = self.sigmoid(z)
        return self.o

    def back_propagation(self, lr, dp_o):
        """
        dp_o  n*1
        :param lr :学习率
        :param dp_o:
        :return:
        """
        dp_z = dp_o * self.o * (1 - self.o)
        self.dp_w = np.dot(dp_z, self.data.T)
        self.dp_b = dp_z
        self.w -= lr * self.dp_w
        self.b -= lr * self.b
        dp_i = np.dot(dp_z.T, self.w)
        return dp_i.T

    # def update_params(self, lr):
    #     """
    #     :param lr: 学习率
    #     :return:
    #     """
    #     self.w -= lr * self.dp_w
    #     self.b -= lr * self.b


if __name__ == '__main__':
    net = BPNet(0.1, 10000)
    data = np.array([[0.05], [0.1]])
    label = np.array([[0.01], [0.99]])
    line = Line(2, 2)
    # line.w = np.array([[0.15, 0.2], [0.25, 0.3]])
    # line.b = np.array([[0.35], [0.35]])
    net.add_layer(line)

    line = Line(2, 2)
    # line.w = np.array([[0.4, 0.45], [0.5, 0.55]])
    # line.b = np.array([[0.6], [0.6]])
    net.add_layer(line)

    train_data={'data':data,'label':label}
    net.train_mode(train_data)
    # res = net.fit(data)
    # res = line.fit(data)
    # print(res)
