from torch.utils.data import DataLoader, random_split
import torchvision

def load_Cifar10_dataset(root, istrain, resize, batch_size, split_ratio=0):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    if istrain:
        datasets = torchvision.datasets.CIFAR10(root=root, train=istrain, download=False, transform=trans)
        total_len = len(datasets)
        train_len, valid_len = int(total_len * split_ratio), total_len - int(total_len * split_ratio)
        datasets = random_split(datasets, [train_len, valid_len])
        return DataLoader(datasets[0], batch_size=batch_size, shuffle=True), DataLoader(datasets[1], batch_size=batch_size, shuffle=True)
    else:
        datasets = torchvision.datasets.CIFAR10(root=root, train=istrain, download=False, transform=trans)
        return DataLoader(datasets, batch_size=batch_size, shuffle=False)