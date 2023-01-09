from get_data_torch import load_Cifar10_dataset
import models
import os
import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(net, train_iter, valid_iter, num_epochs, loss, optim, device, weight_path):
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0
    for epoch in range(num_epochs):
        temp = 0
        num_steps = 0
        true_nums = 0
        total_nums = 0
        # train
        net.train()
        for X, y in tqdm.tqdm(train_iter, f'train epoch{epoch+1}'):
            X, y = X.to(device), y.to(device)
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
            true_nums += (preds == y).cpu().sum()
            
        print(f'epoch: {epoch+1}, loss: {temp / num_steps}, acc: {true_nums / total_nums}')
        train_loss.append(temp / num_steps)
        train_acc.append(true_nums / total_nums)
        # valid
        temp = 0
        num_steps = 0
        true_nums = 0
        total_nums = 0
        net.eval()
        for X, y in tqdm.tqdm(valid_iter, f'valid epoch{epoch+1}'):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = net(X)
                l = loss(y_hat, y)
            
                temp += l.data
                num_steps += 1
                total_nums += y.shape[0]
                preds = nn.Softmax(dim=1)(y_hat)
                preds = torch.argmax(preds, dim=1)
                true_nums += (preds == y).cpu().sum()
            
        print(f'epoch: {epoch+1}, loss: {temp / num_steps}, acc: {true_nums / total_nums}')
        valid_loss.append(temp / num_steps)
        valid_acc.append(true_nums / total_nums)
        if best_acc < valid_acc[-1]:
            print('valid acc improved from %.3f to %.3f' %(best_acc, valid_acc[-1]))
            best_acc = valid_acc[-1]
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'opt_dict': optim.state_dict(),
                }, os.path.join(weight_path, 'epoch-' + str(epoch) + '_acc-%.3f'%(best_acc) + '.pth.tar'))
            print("Save model at {}".format(os.path.join(weight_path, 'epoch-' + str(epoch) + '_acc-%.3f'%(best_acc) + '.pth.tar')))
        else:
            print(f'valid acc did not improve!')
            
    plt.subplot(121)
    plt.plot(range(len(train_loss)), train_loss, label='Train loss')
    plt.plot(range(len(train_loss)), train_acc, label='Train acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(range(len(valid_loss)), valid_loss, label='Valid loss')
    plt.plot(range(len(valid_loss)), valid_acc, label='Valid acc')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = "/content/drive/My Drive/data/"
    weight_path = '/content/drive/My Drive/model/InceptionV3/'
    # file_path = 'data'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size, lr, num_epochs = 64, 0.05, 100
    torch.cuda.empty_cache()
    train_iter, valid_iter = load_Cifar10_dataset(file_path, True, 299, batch_size, split_ratio=0.8)
    
    net = models.get_net('inceptionv3', device)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=net.parameters(), lr=lr)
    net = train(net, train_iter, valid_iter, num_epochs, loss, optim, device, weight_path)