"""
@author:Rai
"""

import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from model import NeuralNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    image = Image.open('./5.png')
    # Transform an image to gray
    image = image.convert('L')
    image = transform(image)
    image = image.view(-1)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)

    net = NeuralNet(ch_input=image.shape[1])
    net.to(device)
    net.load_state_dict(torch.load('./SimpleNet.pth', map_location=device))

    with torch.no_grad():
        output = net(image)
        pred = torch.max(output, dim=1)[1].item()
        print('predict result:', pred)


if __name__ == '__main__':
    main()
