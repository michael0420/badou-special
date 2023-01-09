"""
@author: Rai
推理
"""

from model import LeNet
from torchvision.transforms import transforms
import torch
from PIL import Image


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # [0, 1] -> [-1, -1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet(num_classes=10)
    net.load_state_dict(torch.load('./LeNet.pth', map_location=device))

    img_path = './plane.jpg'
    image = Image.open(img_path)
    image = image.convert(mode='RGB')
    image = transform(image)  # [ch, h, w]
    image = torch.unsqueeze(image, 0)  # [ch, h, w] -> [1, ch, h, w]

    with torch.no_grad():
        output = net(image)
        pred = torch.max(output, dim=1)[1].item()
    print('predict result:', classes[pred])


if __name__ == '__main__':
    main()