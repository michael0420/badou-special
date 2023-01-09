"""
@author: Rai
AlexNet Network
"""

import torch
import torch.nn as nn

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weight=False):
        super().__init__()
        self.featrues = features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

        self.classifier_conv = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(4096, num_classes, kernel_size=1)
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
        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)

        # use Conv2d(512, 4096, 7) replace and flattern() and Linear(512*7*7, 4096)
        x = self.classifier_conv(x)
        x = torch.squeeze(x)

        return x


def make_features(cfg: list):
    layers = []
    in_ch = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_ch,
                               out_channels=v,
                               kernel_size=3,
                               padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_ch = v
    features = nn.Sequential(*layers)
    return features

cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M']
    }

# for debug
def print_shape(model, x):
    output = x
    for i, module in enumerate(model.children()):
        if i == 1:
            output = torch.flatten(output, start_dim=1)
        for layer in module.children():
            output = layer(output)
            print(layer, output.shape)


def vgg(model_name='vgg11', num_classes=1000):
    assert model_name in cfgs, "Warning: {} is not in cfgs dict of vgg models".format(model_name)
    cfg = cfgs[model_name]
    features = make_features(cfg)
    model = VGG(features=features, num_classes=num_classes)

    return model


if __name__ == '__main__':
    net = vgg('vgg11', num_classes=10)
    x = torch.rand(2, 3, 224, 224)
    # print_shape(net, x)
    y = net(x)
    print(y.shape)