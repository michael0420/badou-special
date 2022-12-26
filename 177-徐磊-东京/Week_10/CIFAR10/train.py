"""
@author: Rai
用CIFAR10数据训练神经网络
"""
import torchvision.datasets as datasets
import torchvision.transforms as transfroms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import LeNet


def data_load(batch_size):
    data_transform = {
        'train': transfroms.Compose([
            transfroms.Resize((32, 32)),
            transfroms.RandomHorizontalFlip(),
            transfroms.ColorJitter(),
            # transform data to tensor [0.0,10.]
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transfroms.Compose([
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    root = '../data'
    train_set = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=data_transform['train'])
    test_set = datasets.CIFAR10(root=root, train=False,
                                download=True, transform=data_transform['test'])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class Engine(object):
    def __init__(self, net, batch_size, lr, device):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = device

    def train(self, train_loader):
        train_loss, train_acc = 0, 0
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optim.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optim.step()
            pred = outputs.max(1)[1]

            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()

        data_nums = len(train_loader.dataset)
        train_loss = train_loss * self.batch_size / data_nums
        train_acc = train_acc / data_nums
        return train_loss, train_acc

    def test(self, test_loader):
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = self.criterion(outputs, labels)
                pred = outputs.max(1)[1]

                test_loss += loss.item()
                test_acc += (pred == labels).sum().item()
        data_nums = len(test_loader.dataset)
        test_loss = test_loss * self.batch_size / data_nums
        test_acc = test_acc / data_nums
        return test_loss, test_acc


def show_figure(history, epochs):
    unit = epochs / 10
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
    lr = 0.1
    epochs = 50
    batch_size = 4
    num_classes = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data_load(batch_size)
    net = LeNet(num_classes=num_classes)
    net.to(device)
    data_nums = len(train_loader.dataset)
    model = Engine(net=net,
                   batch_size=batch_size,
                   lr=lr,
                   device=device)
    history = np.zeros((0, 5))
    for epoch in range(epochs):
        train_loss, train_acc = model.train(train_loader)
        test_loss, test_acc = model.test(test_loader)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(
                f'Epoch:[{epoch + 1}/{epochs}], train loss:{train_loss:.5f}, train acc:{train_acc:.4f}, test loss:{test_loss:.5f} test acc:{test_acc:.4f}')
        item = np.array([epoch, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack([history, item])

    save_path = './LeNet.pth'
    torch.save(net.state_dict(), save_path)
    # 绘制训练和验证损失以及精度曲线
    show_figure(history, epochs)


if __name__ == '__main__':
    main()
