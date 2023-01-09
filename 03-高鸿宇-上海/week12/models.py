import torch
import torch.nn as nn
from torchsummary import summary

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class InceptionBlockA(nn.Module):
    def __init__(self, in_channels, fist_block=False) -> None:
        super().__init__()
        if fist_block:
            # branch1
            self.b1_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            #branch2
            self.b2_conv1 = ConvLayer(in_channels, 48, kernel_size=1, stride=1, padding=0)
            self.b2_conv2 = ConvLayer(48, 64, kernel_size=5, stride=1, padding=2)
            #branch3
            self.b3_conv1 = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            self.b3_conv2 = ConvLayer(64, 96, kernel_size=3, stride=1, padding=1)
            self.b3_conv3 = ConvLayer(96, 96, kernel_size=3, stride=1, padding=1)
            #branch4
            self.b4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.b4_conv = ConvLayer(in_channels, 32, kernel_size=1, stride=1, padding=0)
        else:
            # branch1
            self.b1_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            #branch2
            self.b2_conv1 = ConvLayer(in_channels, 48, kernel_size=1, stride=1, padding=0)
            self.b2_conv2 = ConvLayer(48, 64, kernel_size=5, stride=1, padding=2)
            #branch3
            self.b3_conv1 = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
            self.b3_conv2 = ConvLayer(64, 96, kernel_size=3, stride=1, padding=1)
            self.b3_conv3 = ConvLayer(96, 96, kernel_size=3, stride=1, padding=1)
            #branch4
            self.b4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.b4_conv = ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        b1_y = self.b1_conv(x)
        
        b2_y = self.b2_conv1(x)
        b2_y = self.b2_conv2(b2_y)
        
        b3_y = self.b3_conv1(x)
        b3_y = self.b3_conv2(b3_y)
        b3_y = self.b3_conv3(b3_y)
        
        b4_y = self.b4_pool(x)
        b4_y = self.b4_conv(b4_y)
        
        y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        
        return y

class InceptionBlockB(nn.Module):
    def __init__(self, in_channels, block_num=1) -> None:
        super().__init__()
        out_channels = {2:128, 3:160, 4:160, 5:192}
        if block_num == 1:
            # branch1
            self.b1 = nn.Sequential(ConvLayer(in_channels, 384, kernel_size=3, stride=2, padding=0))
            self.b2 = nn.Sequential(ConvLayer(in_channels, 64, kernel_size=1, stride=1, padding=0),
                                    ConvLayer(64, 96, kernel_size=3, stride=1, padding=1),
                                    ConvLayer(96, 96, kernel_size=3, stride=2, padding=0))
            self.b3 = None
            self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        else:
            self.b1 = nn.Sequential(ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0))
            self.b2 = nn.Sequential(ConvLayer(in_channels, out_channels[block_num], kernel_size=1, stride=1, padding=0),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                    ConvLayer(out_channels[block_num], 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
            self.b3 = nn.Sequential(ConvLayer(in_channels, out_channels[block_num], kernel_size=1, stride=1, padding=0),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                    ConvLayer(out_channels[block_num], out_channels[block_num], kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                    ConvLayer(out_channels[block_num], 192, kernel_size=(1, 7), stride=1, padding=(0, 3)))
            self.b4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                    ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        temp = x
        for each_layer in self.b1:
            temp = each_layer(temp)
        b1_y = temp
        temp = x
        for each_layer in self.b2:
            temp = each_layer(temp)
        b2_y = temp
        temp = x
        for each_layer in self.b4:
            temp = each_layer(temp)
        b4_y = temp
        temp = x
        if self.b3 is not None:
            for each_layer in self.b1:
                temp = each_layer(temp)
            b3_y = temp
            y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        else:
            y = torch.concat((b1_y, b2_y, b4_y), dim=1)
        
        return y

class InceptionBlockC_1(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        # branch1
        self.b1_conv1 = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.b1_conv2 = ConvLayer(192, 320, kernel_size=3, stride=2, padding=0)
        
        #branch2
        self.b2_conv1 = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.b2_conv2 = ConvLayer(192, 192, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.b2_conv3 = ConvLayer(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.b2_conv4 = ConvLayer(192, 192, kernel_size=3, stride=2, padding=0)
        
        #branch3
        self.b3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        b1_y = self.b1_conv1(x)
        b1_y = self.b1_conv2(b1_y)
        b2_y = self.b2_conv1(x)
        b2_y = self.b2_conv2(b2_y)
        b2_y = self.b2_conv3(b2_y)
        b2_y = self.b2_conv4(b2_y)
        b3_y = self.b3_pool(x)
        
        y = torch.concat((b1_y, b2_y, b3_y), dim=1)
        
        return y
        
class InceptionBlockC_2(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        # branch1
        self.b1_conv = ConvLayer(in_channels, 320, kernel_size=1, stride=1, padding=0)
        
        #branch2
        self.b2_conv1 = ConvLayer(in_channels, 384, kernel_size=1, stride=1, padding=0)
        self.b2_conv2 = ConvLayer(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.b2_conv3 = ConvLayer(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))
        
        #branch3
        self.b3_conv1 = ConvLayer(in_channels, 448, kernel_size=1, stride=1, padding=0)
        self.b3_conv2 = ConvLayer(448, 384, kernel_size=3, stride=1, padding=1)
        self.b3_conv3 = ConvLayer(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.b3_conv4 = ConvLayer(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))
        
        #branch4
        self.b4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_conv = ConvLayer(in_channels, 192, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        b1_y = self.b1_conv(x)
        b2_y = self.b2_conv1(x)
        b2_y1 = self.b2_conv2(b2_y)
        b2_y2 = self.b2_conv3(b2_y)
        b2_y = torch.concat((b2_y1, b2_y2), dim=1)
        b3_y = self.b3_conv1(x)
        b3_y = self.b3_conv2(b3_y)
        b3_y1 = self.b3_conv3(b3_y)
        b3_y2 = self.b3_conv4(b3_y)
        b3_y = torch.concat((b3_y1, b3_y2), dim=1)
        b4_y = self.b4_pool(x)
        b4_y = self.b4_conv(b4_y)
        
        y = torch.concat((b1_y, b2_y, b3_y, b4_y), dim=1)
        
        return y

class ConvDwLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation1 = nn.ReLU6()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU6()
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.activation2(x)
        return x

class Inceptionv3(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4 = ConvLayer(64, 80, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(80, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.inception_block_a1 = InceptionBlockA(192, True)
        self.inception_block_a2 = InceptionBlockA(256, False)
        self.inception_block_a3 = InceptionBlockA(288, False)

        self.inception_block_b1 = InceptionBlockB(288, 1)
        self.inception_block_b2 = InceptionBlockB(768, 2)
        self.inception_block_b3 = InceptionBlockB(768, 3)
        self.inception_block_b4 = InceptionBlockB(768, 4)
        self.inception_block_b5 = InceptionBlockB(768, 5)
        
        self.inception_block_c1 = InceptionBlockC_1(768)
        self.inception_block_c2 = InceptionBlockC_2(1280)
        self.inception_block_c3 = InceptionBlockC_2(2048)
        
        self.pool3 = nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout2d()
        self.conv6 = ConvLayer(2048, 10, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.inception_block_a1(x)
        x = self.inception_block_a2(x)
        x = self.inception_block_a3(x)
        x = self.inception_block_b1(x)
        x = self.inception_block_b2(x)
        x = self.inception_block_b3(x)
        x = self.inception_block_b4(x)
        x = self.inception_block_b5(x)
        x = self.inception_block_c1(x)
        x = self.inception_block_c2(x)
        x = self.inception_block_c3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.softmax(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 3, stride=2, padding=1)
        self.convdw1 = ConvDwLayer(32, 32, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(32, 64, 1, stride=1, padding=0)
        self.convdw2 = ConvDwLayer(64, 64, 3, stride=2, padding=1)
        self.conv3 = ConvLayer(64, 128, 1, stride=1, padding=0)
        self.convdw3 = ConvDwLayer(128, 128, 3, stride=1, padding=1)
        self.conv4 = ConvLayer(128, 128, 1, stride=1, padding=0)
        self.convdw4 = ConvDwLayer(128, 128, 3, stride=2, padding=1)
        self.conv5 = ConvLayer(128, 256, 1, stride=1, padding=0)
        self.convdw5 = ConvDwLayer(256, 256, 3, stride=1, padding=1)
        self.conv6 = ConvLayer(256, 256, 1, stride=1, padding=0)
        self.convdw6 = ConvDwLayer(256, 256, 3, stride=2, padding=1)
        self.conv7 = ConvLayer(256, 512, 1, stride=1, padding=0)
        
        # 5xconvdw 5xconv
        self.convdw7 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv8 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw8 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv9 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw9 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv10 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw10 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv11 = ConvLayer(512, 512, 1, stride=1, padding=0)
        self.convdw11 = ConvDwLayer(512, 512, 3, stride=1, padding=1)
        self.conv12 = ConvLayer(512, 512, 1, stride=1, padding=0)
        
        self.convdw12 = ConvDwLayer(512, 512, 3, stride=2, padding=1)
        self.conv13 = ConvLayer(512, 1024, 1, stride=1, padding=0)
        self.convdw13 = ConvDwLayer(1024, 1024, 3, stride=1, padding=1)
        self.conv14 = ConvLayer(1024, 1024, 1, stride=1, padding=0)
        
        # classification layers
        self.pooling = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.convdw1(x)
        x = self.conv2(x)
        x = self.convdw2(x)
        x = self.conv3(x)
        x = self.convdw3(x)
        x = self.conv4(x)
        x = self.convdw4(x)
        x = self.conv5(x)
        x = self.convdw5(x)
        x = self.conv6(x)
        x = self.convdw6(x)
        x = self.conv7(x)
        x = self.convdw7(x)
        x = self.conv8(x)
        x = self.convdw8(x)
        x = self.conv9(x)
        x = self.convdw9(x)
        x = self.conv10(x)
        x = self.convdw10(x)
        x = self.conv11(x)
        x = self.convdw11(x)
        x = self.conv12(x)
        x = self.convdw12(x)
        x = self.conv13(x)
        x = self.convdw13(x)
        x = self.conv14(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def get_net(net_name, device, weight_to_load=None):
    if net_name == 'inceptionv3':
        net = Inceptionv3(device)
        summary(net.cuda(), (3, 299, 299))
    if net_name == 'mobile_net':
        net = MobileNet(device)
        summary(net.cuda(), (3, 224, 224))
    print(net)
    net.to(device)
    net.apply(xavier_init_weights)
    if weight_to_load:
        checkpoint = torch.load(weight_to_load)
        net.load_state_dict(checkpoint['state_dict'])
    return net

if __name__ == "__main__":
    net = get_net('inceptionv3', torch.device('cpu'))
    # summary(net, (3, 224, 224))