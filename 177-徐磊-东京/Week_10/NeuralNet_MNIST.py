"""
@author: Rai
训练MNIST数据集网络
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

# 数据读取及预处理
def data_load(batch_size):
    # import torchvision.datasets as datasets
    # import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 数据的范围[0,1] -> [-1,1]
        transforms.Normalize(0.5, 0.5),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    data_root = './data'
    train_set = datasets.MNIST(root=data_root, train=True,
                               download=True, transform=transform)

    test_set = datasets.MNIST(root=data_root, train=False,
                              download=False, transform=transform)

    # from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    num_classes = 10
    return train_set, test_set, train_loader, test_loader, num_classes

# 定义神经网络
class NeuralNet(nn.Module):
    def __init__(self, ch_input, ch_output, ch_hidden):
        super().__init__()
        self.fc1 = nn.Linear(ch_input, ch_hidden)
        self.fc2 = nn.Linear(ch_hidden, ch_hidden)
        self.fc3 = nn.Linear(ch_hidden, ch_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)

        return x5

# 定义训练和验证模型
class Model(object):
    def __init__(self, net, cost, optimizer, lr, device):
        self.net = net
        self.cost = self.create_loss(cost)
        self.optim = self.create_optim(optimizer, lr)
        self.device = device

    def create_loss(self, cost):
        f_cost_dict = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MST': nn.MSELoss()
        }
        return f_cost_dict[cost]

    def create_optim(self, optimizer, lr, **kwargs):
        f_optim_dict = {
            'SGD': optim.SGD(self.net.parameters(), lr=lr, **kwargs),
            'ADAM': optim.Adam(self.net.parameters(), lr=lr, **kwargs),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=lr, **kwargs)
        }
        return f_optim_dict[optimizer]

    def train(self, train_loader, batch_size, data_nums):
        train_acc, train_loss= 0, 0
        for inputs, labels in train_loader:
            # 数据写入GPU或CPU
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # 预测值
            outputs = self.net(inputs)
            # 损失
            loss = self.cost(outputs, labels)
            # 防止梯度出现累加
            self.optim.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            self.optim.step()
            # 概率最大的即为预测值.max返回（最大值，索引）
            pred = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()

        train_acc = train_acc / data_nums
        train_loss = train_loss * batch_size / data_nums
        return train_acc, train_loss

    def test(self, test_loader, batch_size, data_nums):
        test_acc, test_loss = 0, 0
        # 关闭梯度，节省内存
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                pred = torch.max(outputs, 1)[1]
                test_loss += loss.item()
                test_acc += (pred == labels).sum().item()

        test_acc = test_acc / data_nums
        test_loss = test_loss * batch_size / data_nums
        return test_acc, test_loss

def show_figure(history, epochs):
    unit = epochs / 2
    # 绘制损失曲线
    plt.plot(history[:, 0], history[:, 1], 'b', label='train_loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='test_loss')
    plt.title('loss curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('损失曲线.png')
    plt.show()

    # 绘制精度曲线
    plt.plot(history[:, 0], history[:, 2], 'b', label='train_acc')
    plt.plot(history[:, 0], history[:, 4], 'g', label='test_acc')
    plt.title('acc curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig('精度曲线')
    plt.show()


def main():
    batch_size = 128
    lr = 0.01
    epochs = 50
    # 设备支持GPU使用GPU训练，反之使用CPU训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set, test_set, train_loader, test_loader, num_classes = data_load(batch_size)
    data_nums = len(train_loader.dataset)
    image, lable = train_set[0]
    ch_input = image.shape[0]
    net = NeuralNet(ch_input=ch_input, ch_output=num_classes, ch_hidden=128).to(device)
    model = Model(net=net, cost='CROSS_ENTROPY', optimizer='SGD', lr=lr, device=device)
    # 保存训练和验证的损失及精度结果
    history = np.zeros((0, 5))

    for epoch in range(epochs):
        train_acc, train_loss = model.train(train_loader, batch_size, data_nums)
        test_acc, test_loss = model.test(test_loader, batch_size, data_nums)
        print(
            f'Epoch:[{epoch + 1}/{epochs}], train_loss:{train_loss:.5f}, train_acc:{train_acc:.4f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.4f}')
        item = np.array([epoch + 1, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack([history, item])

    # 绘制训练和验证损失以及精度曲线
    show_figure(history, epochs)

if __name__ == '__main__':
    main()
