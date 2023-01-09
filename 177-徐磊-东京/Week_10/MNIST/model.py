"""
@author: Rai
定义一个简单神经网络
"""
import torch.nn as nn


# 定义神经网络
class NeuralNet(nn.Module):
    def __init__(self, ch_input=1, ch_output=10, ch_hidden=128):
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