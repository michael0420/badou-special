import numpy as np
import torch
import time
from alexnet import AlexNet
import torch.nn as nn
from data_loader import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform1 = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def evaluate(model, tests, labels):
    with torch.no_grad():
        outputs = model(tests)
        predicted = torch.argmax(outputs, 1)
        total = labels.shape[0]
        correct = torch.sum(predicted == labels)
    return correct / total

def main():
    epochs = 10             #训练轮数
    learning_rate = 0.001     # 学习率
    model = AlexNet()     # 建立模型
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    print('Start Training...')
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        watch_loss = []
        test_acc = []
        for inputs, targets in train_loader:
            targets = nn.functional.one_hot(targets).float()
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, targets)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print('Evaluating ...')
        model.eval()
        for tests, targets in test_loader:
            acc = evaluate(model, tests, targets)
            test_acc.append(acc)
        print('Accuracy of the network on the test images: {:.1f}%'.format(np.mean(test_acc) * 100))

        stop_time = time.time()
        print('time is:{:.4f}s'.format(stop_time - start_time))
    torch.save(model.state_dict(), "Alexnet_model.pth")

if __name__ == '__main__':
    train_loader = DataLoader(MyDataset(r'.\image_dataset\train_dataset.txt', r".\image", transform1), batch_size=100, shuffle=True)
    test_loader = DataLoader(MyDataset(r'.\image_dataset\test_dataset.txt', r".\test_image", transform2), batch_size=100, shuffle=False)
    main()
