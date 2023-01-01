"""
@author: Rai
推理
"""

from model import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from PIL import Image


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = resnet_18(num_classes=10)

    net.load_state_dict(torch.load('./resnet18_finetuning.pth', map_location=device))
    net.to(device)

    img_path = '../data/car.jpg'
    image = Image.open(img_path)
    image = image.convert(mode='RGB')
    image = transform(image)  # [ch, h, w]
    image = torch.unsqueeze(image, 0)  # [ch, h, w] -> [1, ch, h, w]

    # prediction
    net.eval()
    with torch.no_grad():
        image = image.to(device)
        output = net(image)
        print(output)
        pred = torch.max(output, dim=1)[1].item()
    print('predict result:', classes[pred])


if __name__ == '__main__':
    main()