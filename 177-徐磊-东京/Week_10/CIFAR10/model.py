"""
@author: Rai
LeNet Network
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(x1)
        x3 = self.pool1(x2)

        x4 = self.conv2(x3)
        x5 = torch.sigmoid(x4)
        x6 = self.pool2(x5)

        x7 = self.flatten(x6)
        x8 = self.fc1(x7)
        x9 = torch.sigmoid(x8)

        x10 = self.fc2(x9)
        x11 = torch.sigmoid(x10)
        x12 = self.fc3(x11)
        return x12


if __name__ == '__main__':
    net = LeNet(10)
    x = torch.rand(2, 3, 32, 32)
    y = net(x)
    print(y.shape)