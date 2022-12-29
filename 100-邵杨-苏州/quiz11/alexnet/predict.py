from PIL import Image
import numpy as np
from alexnet import AlexNet
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
classes = {0: '猫', 1: '狗'}

def predict(target):
    model = AlexNet()
    model.load_state_dict(torch.load("Alexnet_model.pth"))
    model.eval()  # 测试模式
    with torch.no_grad():
        output = model(target)
    return output

if __name__ == '__main__':
    img = Image.open("./Test.jpg").convert('RGB')
    plt.imshow(img)
    plt.show()
    img = transform(img)
    #print(img.shape)
    img = img.unsqueeze(0)
    predict = predict(img)
    predict_class = classes[predict.argmax(dim=1).item()]
    print('预测猫的概率为:{:.2f}%， 狗的概率为：{:.2f}%'.format(predict[0][0] * 100, predict[0][1]*100))
    print("预测类别：", predict_class)




