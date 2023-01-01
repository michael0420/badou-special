"""
@author: Rai
AlexNet Network
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weight=False):
        super().__init__()
        self.featrues = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # [N, 3, 224, 224] -> [N, 96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                      # output:[N, 96, 27, 27]

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),      # output:[N, 256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                      # output:[N, 256, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),     # output:[N, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),     # output:[N, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),     # output:[N, 256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                      # output:[N, 256, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

        if init_weight:
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant(layer.bias, 0)
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant(layer.bias, 0)

    def forward(self, x):
        x = self.featrues(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def print_shape(model, x):
    output = x
    for i, module in enumerate(model.children()):
        if i == 1:
            output = torch.flatten(output, start_dim=1)
        for layer in module.children():
            output = layer(output)
            print(layer, output.shape)


if __name__ == '__main__':
    net = AlexNet(10)
    x = torch.rand(2, 3, 224, 224)
    print_shape(net, x)
    y = net(x)
    print(y.shape)