import torch
import torch.nn as nn
import tqdm
import models
import pandas as pd
from get_data_torch import load_Cifar10_dataset

def predict(net, test_iter, device):
    net.eval()
    preds_labels = torch.tensor([])
    gt = torch.tensor([])
    true_nums, total_nums = 0, 0
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            total_nums += y.shape[0]
            preds = nn.Softmax(dim=1)(y_hat)
            preds = torch.argmax(preds, dim=1)
            true_nums += (preds == y).cpu().sum()
            preds_labels = torch.cat((preds_labels, preds.cpu()))
            gt = torch.cat((gt, y.cpu()))
    print(f'acc: {true_nums / total_nums}')
    return preds_labels, gt

if __name__ == "__main__":
    file_path = "/content/drive/My Drive/data/"
    weight_to_load = '/content/drive/My Drive/model/Mobilenet/epoch-26_acc-0.717.pth.tar'
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = models.get_net('inceptionv3', device, weight_to_load)
    print(net)
    test_iter = load_Cifar10_dataset(file_path, False, 299, batch_size)
    preds, gt = predict(net, test_iter, device)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels = [classes[int(i.data)] for i in preds]
    gt = [classes[int(i.data)] for i in gt]
    df = pd.DataFrame({'labels': labels, 'gt':gt})
    print(df)