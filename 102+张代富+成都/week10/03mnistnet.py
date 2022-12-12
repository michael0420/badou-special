import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    @staticmethod
    def create_cost(cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost.get(cost)

    def create_optimizer(self, optimist, **rests):
        support_optimizer = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optimizer.get(optimist)

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print(
                        f'[epoch {epoch + 1}, {(i + 1) / len(train_loader) * 100:.2f}%] loss: {running_loss / 100:.2f}')
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0, ], [1, ])
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    return train_loader, test_loader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    _net = MnistNet()
    model = Model(_net, 'CROSS_ENTROPY', 'RMSP')
    train, test = load_data()
    model.train(train)
    model.evaluate(test)
