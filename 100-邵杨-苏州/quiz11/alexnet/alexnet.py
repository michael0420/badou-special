import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        # [(w-f+2p)/s+1]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(9, 9), stride=(4, 4), padding=1),   # [batch,3, 224, 224]-> [batch, 96, 55, 55]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [b, 96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),  # [b, 256, 27, 27]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [b, 256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [b, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [b, 384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [b, 256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   # [b, 256, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(100, num_classes),
            nn.Softmax(dim=1))

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y_pred = self.classifier(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred
