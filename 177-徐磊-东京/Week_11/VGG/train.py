"""
@author: Rai
用CIFAR10数据训练神经网络
"""
import os

import torchvision.datasets as datasets
import torchvision.transforms as transfroms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import vgg
from tqdm import tqdm
import sys
import wandb
import datetime,time

def data_load(batch_size):
    data_transform = {
        'train': transfroms.Compose([
            transfroms.RandomHorizontalFlip(),
            transfroms.RandomResizedCrop(224),
            # transform data to tensor [0.0,10.]
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transfroms.Compose([
            transfroms.Resize((224, 224)),
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    root = '../../data'
    train_set = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=data_transform['train'])
    test_set = datasets.CIFAR10(root=root, train=False,
                                download=True, transform=data_transform['test'])

    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_num, test_num = len(train_set), len(test_set)
    print('using {} images for train, {} images for test'.format(train_num, test_num))
    return train_loader, test_loader


class Engine(object):
    def __init__(self, net, batch_size, lr, device):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.Adam(net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = device

    def train(self, train_loader):
        self.net.train()
        train_loss, train_acc = 0, 0
        # train_bar = tqdm(train_loader, file=sys.stdout, desc='train')
        # for images, labels in train_bar:
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
        self.net.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            # test_bar = tqdm(test_loader, file=sys.stdout, desc='test')
            # for images, labels in test_bar:
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
    fig1 = plt.figure()
    plt.plot(history[:, 0], history[:, 1], 'b', label='train_loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='test_loss')
    plt.title('loss curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig1.savefig('损失曲线.png')
    # plt.show()

    # 绘制精度曲线
    fig2 = plt.figure()
    plt.plot(history[:, 0], history[:, 2], 'b', label='train_acc')
    plt.plot(history[:, 0], history[:, 4], 'g', label='test_acc')
    plt.title('acc curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('acc')
    fig2.savefig('精度曲线')
    # plt.show()

def create_log(log_name, model, epochs, batch_size):
    config = {
        "log_name:": log_name,
        'model': model,
        "epoches": epochs,
        "batch_size": batch_size
    }

def main():
    dt_end = datetime.datetime.now()
    time_run_start = dt_end.strftime('%Y%m%d_%H%M%S')
    model_name = 'VGG'
    print("start_time:"+time_run_start+'\n')
    log_name = str(time_run_start[2:]) + '_' + model_name
    
    lr = 0.0001
    epochs = 1
    batch_size = 128
    num_classes = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data_load(batch_size)
    net = vgg(model_name='vgg11', num_classes=num_classes)
    net.to(device)
    data_nums = len(train_loader.dataset)
    model = Engine(net=net,
                   batch_size=batch_size,
                   lr=lr,
                   device=device)
    history = np.zeros((0, 5))
    save_path = './VGG.pth'
    best_acc = 0.0

    # wandb configration
    config = {
        "log_name:": log_name,
        'model': model_name,
        "epoches": epochs,
        "batch_size": batch_size,
        'lr': lr
    }

    wandb.init(project="Exercise",
                    name=log_name, 
                    id=wandb.util.generate_id(), 
                    config=config,
                    save_code=True)

    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = model.train(train_loader)
        test_loss, test_acc = model.test(test_loader)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(
                'Epoch:[{:3d}/{}, train loss:{:.5f}, train acc:{:.4f}, test loss:{:.5f} test acc:{:.4f}'.format(
                    epoch + 1, epochs, train_loss,train_acc, test_loss, test_acc))
        item = np.array([epoch, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack([history, item])

        wandb.log({"Train_loss":train_loss, "Test_loss":test_loss}, step=epoch)
        wandb.log({"Train_acc":train_acc, "Test_acc":test_acc}, step=epoch)
        if test_acc > best_acc:
            torch.save(net.state_dict(), save_path)
    # 绘制训练和验证损失以及精度曲线
    show_figure(history, epochs)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    dt_end = datetime.datetime.now()
    time_run_end = dt_end.strftime('%Y%m%d_%H_%M_%S')
    print("time end:"+time_run_end+'\n')
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    main()
