import time
import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn


transform1 = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform2 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_train = datasets.CIFAR10('./c10data', train=True, transform=transform1,
                         download=True)
data_test = datasets.CIFAR10('./c10data', train=False, transform=transform2,
                         download=False)
train_loader = DataLoader(data_train, batch_size=250, shuffle=False)
train_test = DataLoader(data_test, batch_size=250, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # [(w-f+2p)/s+1]
        # [batch, 32, 32, 3]-> [batch, 32, 32, 64]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2) #[b, 16, 16, 64]
        )
        #[b, 16, 16, 64]->[b, 16, 16, 128]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2) #[b, 8, 8, 128]
        )
        # [b, 8, 8, 128]->[b, 8, 8, 256]
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2) #[b, 4, 4, 256]
        )

        self.dense = nn.Sequential(
            nn.Linear(4*4*256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        y_pred = self.dense(x)
        if y is not None:
            return self.loss(y_pred, y)      #预测值和真实值计算损失
        else:
            return y_pred

def evaluate(model, tests, labels):
    total, correct = 0, 0
    with torch.no_grad():  # no grad when test and predict
        outputs = model(tests)
        predicted = torch.argmax(outputs, 1)
        total = labels.shape[0]
        correct = torch.sum(predicted == labels)
    return correct / total

def main():
    epochs = 10              #训练轮数
    learning_rate = 0.001     # 学习率
    model = CNN()     # 建立模型
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    # 训练过程
    print('Start Training...')
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        watch_loss = []
        test_acc = []
        for inputs, targets in train_loader:
            targets = nn.functional.one_hot(targets).float()
            inputs, targets = inputs.to(device), targets.to(device)
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, targets)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print('Evaluating ...')
        model.eval()
        for tests, targets in train_test:
            tests, targets = tests.to(device), targets.to(device)
            acc = evaluate(model, tests, targets)
            test_acc.append(acc)
        print('Accuracy of the network on the test images: {:.1f}%'.format(np.mean(test_acc)*100))

        stop_time = time.time()
        print('time is:{:.4f}s'.format(stop_time-start_time))

if __name__ == '__main__':
    main()
    print('===Finish Training===')