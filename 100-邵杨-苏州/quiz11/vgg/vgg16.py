import torch.nn as nn

class vgg16(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg16, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 3, 224, 224]-> [batch, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1), #[batch, 64, 224, 224]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)) #[batch, 64, 112, 112]

        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1), #[batch, 128, 112, 112]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)) #[batch, 128, 56, 56]

        self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1), #[batch, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 256, 56, 56]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)) #[batch, 256, 28, 28]

        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), #[batch, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 512, 28, 28]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)) #[batch, 512, 14, 14]

        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 512, 14, 14]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), #[batch, 512, 14, 14]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [batch, 512, 14, 14]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)) #[batch, 512, 7, 7]

        self.layer6 = nn.Sequential(
                nn.Conv2d(512, 4096, kernel_size=(7, 7), stride=(1, 1), padding=0),  # [batch, 4096, 1, 1]
                nn.ReLU(inplace=True))

        self.layer7 = nn.Sequential(
                nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1), padding=0),  # [batch, 4096, 1, 1]
                nn.ReLU(inplace=True))

        self.layer8 = nn.Conv2d(4096, 1000, kernel_size=(1, 1), stride=(1, 1), padding=0)  # [batch, 1000, 1, 1]

        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size(0), -1)
        y_pred = self.softmax(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#model = vgg16()
#print(model)


