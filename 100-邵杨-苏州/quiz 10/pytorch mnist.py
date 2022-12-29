import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import mnist



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]) #(x-mean)/std
data_train = mnist.MNIST('./data', train=True, transform=transform,
                         target_transform=None, download=True)
data_test = mnist.MNIST('./data', train=False, transform=transform,
                         target_transform=None, download=False)
train_loader = DataLoader(data_train, batch_size=300, shuffle=True)
test_loader = DataLoader(data_train, batch_size=300, shuffle=False)



class TorchModel(nn.Module):
    def __init__(self, i_nodes, h_nodes, o_nodes):
        super(TorchModel, self).__init__()
        self.classify_1 = nn.Linear(i_nodes, h_nodes, bias=False)     #线性层
        self.classify_2 = nn.Linear(h_nodes, o_nodes, bias=False)
        self.activation_1 = nn.ReLU()                         #relu函数
        self.activation_2 = nn.Softmax(dim=1)                 #softmax函数
        self.loss = nn.CrossEntropyLoss()                #loss函数采用交叉熵损失

    def forward(self, x, y=None):            #当输入真实标签，返回loss值；无真实标签，返回预测值
        x = x.view(x.size(0), -1)
        #nn.Dropout(p=0.3)
        hidden_input = self.classify_1(x)
        hidden_output = self.activation_1(hidden_input)
        out_input = self.classify_2(hidden_output)
        out_output = self.activation_1(out_input)
        y_pred = self.activation_2(out_output)   #(batch size,  10)
        if y is not None:
            return self.loss(y_pred, y)      #预测值和真实值计算损失
        else:
            return y_pred

def evaluate(model, images, labels):
    total, correct = 0, 0
    with torch.no_grad():  # no grad when test and predict
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)
        total = labels.shape[0]
        correct = torch.sum(predicted == labels)
    return correct / total

def main():
    epochs = 15              #训练轮数
    learning_rate = 0.0008     # 学习率
    model = TorchModel(784, 289, 10)     # 建立模型
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    print('Start Training...')
    # 训练过程
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        test_acc = []
        for inputs, targets in train_loader:
            targets = nn.functional.one_hot(targets).float()
            optim.zero_grad()   # 梯度归零
            loss = model(inputs, targets)  # 计算loss
            loss.backward()   # 计算梯度
            optim.step()     # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print('Evaluating ...')
        model.eval()
        for tests, targets in test_loader:
            acc = evaluate(model, tests, targets)
            test_acc.append(acc)
        print('Accuracy of the network on the test images: {:.1f}%'.format(np.mean(test_acc) * 100))


if __name__ == '__main__':
    main()
    print('===Finish Training===')

