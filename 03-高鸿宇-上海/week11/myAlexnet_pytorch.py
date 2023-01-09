import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

class My_Alexnet(nn.Module):
    def __init__(self) -> None:
        super(My_Alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 1)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxPool3 = nn.MaxPool2d(3, stride=2)
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(5*5*256, 4096), nn.ReLU(),
                                 nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxPool3(x)
        for layer in self.cls:
            x = layer(x)
        return x

def load_dataset(istrain, batch_size, resize=None):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    datasets = torchvision.datasets.CIFAR10(root=r'data', train=istrain, download=False, transform=trans)
    
    return DataLoader(datasets, batch_size=batch_size, shuffle=istrain)

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)
        
def train(net, train_iter, num_epochs, loss, optim, weight_dir):
    train_loss = []
    train_acc = []
    min_loss = torch.inf
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
        if(min_loss > temp / num_steps):
            print(f'loss of best model has been changed from {min_loss} to {temp / num_steps}')
            min_loss = temp / num_steps
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'opt_dict': optim.state_dict(),
                }, os.path.join(weight_dir, 'Alexnet_epoch-' + str(epoch) + f'_loss-{min_loss}' + '.pth.tar'))
            print("Save model at {}".format(os.path.join(weight_dir, 'Alexnet_epoch-' + str(epoch) + f'_loss-{min_loss}' + '.pth.tar')))
        train_loss.append(temp / num_steps)
        train_acc.append(true_nums / total_nums)
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='Train loss')
    plt.plot(range(len(train_loss)), train_acc, label='Train acc')
    plt.legend()
    plt.show()

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
    weight_dir = r'week11\model\alexnet'
    model_to_load = r''
    train_iter = load_dataset(True, batch_size,resize=(224,224))
    test_iter = load_dataset(False, batch_size,resize=(224,224))
    
    net = My_Alexnet()
    print(net)
    
    net.apply(xavier_init_weights)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=net.parameters(), lr=lr)
    
    train(net, train_iter, num_epochs, loss, optim, weight_dir)
    
    ################################################################################
    # net = My_Alexnet()
    # checkpoint = torch.load(model_to_load)
    # net.load_state_dict(checkpoint['state_dict'])
    # preds = predict(net, test_iter)
    # classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # labels = [classes[int(i.data)] for i in preds]
    # df = pd.DataFrame({'labels': labels})
    # print(df)