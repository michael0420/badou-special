import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class My_Cnn(nn.Module):
    def __init__(self) -> None:
        super(My_Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2)
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(8*8*64, 384), nn.ReLU(),
                                 nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        for layer in self.cls:
            x = layer(x)
        return x

def load_dataset(istrain, batch_size):
    trans = torchvision.transforms.ToTensor()
    datasets = torchvision.datasets.CIFAR10(root=r'data', train=istrain, download=False, transform=trans)
    
    return DataLoader(datasets, batch_size=batch_size, shuffle=istrain)

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)

def train(net, train_iter, num_epochs, loss, optim):
    train_loss = []
    train_acc = []
    for epoch in range(num_epochs):
        temp = 0
        num_steps = 0
        true_nums = 0
        total_nums = 0
        for X, y in tqdm.tqdm(train_iter):
            y_hat = net(X)
            l = loss(y_hat, y)
            optim.zero_grad()
            l.backward()
            optim.step()
            
            temp += l.data
            num_steps += 1
            total_nums += y.shape[0]
            preds = nn.Softmax(dim=1)(y_hat)
            preds = torch.argmax(preds, dim=1)
            true_nums += (preds == y).sum()
            
        print(f'epoch: {epoch+1}, loss: {temp / num_steps}, acc: {true_nums / total_nums}')
        train_loss.append(temp / num_steps)
        train_acc.append(true_nums / total_nums)
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='Train loss')
    plt.plot(range(len(train_loss)), train_acc, label='Train acc')
    plt.legend()
    plt.show()
    return net

def predict(net, test_iter):
    net.eval()
    preds_labels = torch.tensor([])
    true_nums, total_nums = 0, 0
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_iter):
            y_hat = net(X)
            total_nums += y.shape[0]
            preds = nn.Softmax(dim=1)(y_hat)
            preds = torch.argmax(preds, dim=1)
            true_nums += (preds == y).sum()
            preds_labels = torch.cat((preds_labels, preds))
    print(f'acc: {true_nums / total_nums}')
    return preds_labels

if __name__ == "__main__":
    batch_size, lr, num_epochs = 128, 0.1, 15
    
    train_iter = load_dataset(True, batch_size)
    test_iter = load_dataset(False, batch_size)
    
    net = My_Cnn()
    print(net)
    
    net.apply(xavier_init_weights)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=net.parameters(), lr=lr)
    
    net = train(net, train_iter, num_epochs, loss, optim)
    preds = predict(net, test_iter)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels = [classes[int(i.data)] for i in preds]
    df = pd.DataFrame({'labels': labels})
    print(df)