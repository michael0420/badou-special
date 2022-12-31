import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import tqdm

def load_dataset(istrain, batch_size):
    trans = torchvision.transforms.ToTensor()
    datasets = torchvision.datasets.MNIST(root=r'data\Mnist', train=istrain, download=False, transform=trans)
    
    return DataLoader(datasets, batch_size=batch_size, shuffle=istrain)

class My_Ann(nn.Module):
    def __init__(self, num_inputs, nun_hidden, num_outputs) -> None:
        super(My_Ann, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_inputs, nun_hidden)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(nun_hidden, num_outputs)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

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
    num_inputs, nun_hidden, num_outputs = 28*28, 512, 10
    batch_size, lr, num_epochs = 16, 0.01, 10
    
    train_iter = load_dataset(True, batch_size)
    test_iter = load_dataset(False, batch_size)
    
    net = My_Ann(num_inputs, nun_hidden, num_outputs)
    print(net)
    
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=net.parameters(), lr=lr)
    
    net = train(net, train_iter, num_epochs, loss, optim)
    preds = predict(net, test_iter)