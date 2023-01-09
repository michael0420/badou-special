"""
@author:Rai
Resnet Neural Network
"""

import torch
import torch.nn as nn


# for 18, 34 layer
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += identity

        out = self.relu(x)

        return out


# for 50, 101, 152 layer
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x += identity
        out = self.relu(x)

        return out


class Resnet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_ch = 64

        self.layer1 = self.make_layer(block, ch=64, block_num=block_nums[0], stride=1)
        self.layer2 = self.make_layer(block, ch=128, block_num=block_nums[1], stride=2)
        self.layer3 = self.make_layer(block, ch=256, block_num=block_nums[2], stride=2)
        self.layer4 = self.make_layer(block, ch=512, block_num=block_nums[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, ch, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_ch, out_channels=ch * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch * block.expansion)
            )

        layers = []
        layers.append(
            block(in_ch=self.in_ch, out_ch=ch, stride=stride, downsample=downsample)
        )

        self.in_ch = ch * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(in_ch=self.in_ch, out_ch=ch, stride=1, downsample=None)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def resnet_18(num_classes=1000):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return Resnet(block=BasicBlock, block_nums=[2, 2, 2, 2], num_classes=num_classes)


def resnet_34(num_classes=1000):
    return Resnet(block=BasicBlock, block_nums=[3, 4, 6, 3], num_classes=num_classes)


def resnet_50(num_classes=1000):
    return Resnet(block=BasicBlock, block_nums=[3, 4, 6, 3], num_classes=num_classes)


def resnet_101(num_classes=1000):
    return Resnet(block=BasicBlock, block_nums=[3, 8, 23, 3], num_classes=num_classes)


def resnet_152(num_classes=1000):
    return Resnet(block=BasicBlock, block_nums=[3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    net = resnet_18()
    x = torch.rand(2, 3, 224, 224)
    y = net(x)
    print(y.shape)
