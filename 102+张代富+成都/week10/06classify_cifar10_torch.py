import torch
import torch.nn as nn
from torchvision.datasets import cifar
from torchvision import transforms
from torch.utils.data import DataLoader


class AlexNet(nn.Module):
    def __init__(self, batch_size, class_num):
        super().__init__()
        self.batch_size = batch_size
        self.path = r'C:\Users\daifu\.keras\datasets'
        self.transforms = transforms.Compose([
            transforms.CenterCrop([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train = cifar.CIFAR10(root=self.path, train=True, transform=self.transforms, download=True)
        self.train_loader = DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)
        self.test = cifar.CIFAR10(root=self.path, train=False, transform=self.transforms)
        self.test_loader = DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 192, (3, 3), (1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, (3, 3), (1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.75),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

        self._optimizer = None
        self._criterion = None
        self.train_loss = 0
        self.train_acc = 0

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], 256 * 4 * 4)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = AlexNet(512, 10)
    lr = 1e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(1):
        n = 0
        train_loss = 0
        train_acc = 0
        for data, label in model.train_loader:
            n += 1
            # 前向传播
            output = model(data)
            # 记录单批次一次batch的loss
            loss = criterion(output, label)
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化
            # 累计单批次误差
            train_loss = train_loss + loss.item()
            # 计算分类的准确率
            _, pred = output.max(1)  # 求出每行的最大值,值与序号pred
            num_correct = (pred == label).sum().item()
            acc = num_correct / label.shape[0]
            train_acc = train_acc + acc

            print('epoch: {}, trainloss: {:.4f},trainacc: {:.4f}'.format(i + 1, train_loss / len(model.train_loader), train_acc / len(model.train_loader)))

    # 测试集进行测试
    test_loss = 0
    eval_acc = 0
    model.eval()
    for data, label in model.test_loader:
        output = model(data)
        # 记录单批次一次batch的loss，并且测试集不需要反向传播更新网络
        loss = criterion(output, label)
        test_loss = test_loss + loss.item()
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / label.shape[0]
        eval_acc = eval_acc + acc

    print('test  loss: {:.4f},acc: {:.4f}'.format(test_loss / len(model.test_loader), eval_acc / len(model.test_loader)))
