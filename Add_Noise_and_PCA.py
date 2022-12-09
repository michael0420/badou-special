import numpy as np
import cv2
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt

def Add_GaussianNoise(image, mu, sigma):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise =cv2.add(image,noise)

    return gauss_noise

def Add_PepperSaltNoise(image,n):
    new_image = image.copy()
    w, h = image.shape[:2]
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            new_image[x, y] = 0# 椒
        else:
            new_image[x, y] = 255# 盐
    return new_image

def Decomposion():
    x, y = load_iris(return_X_y=True)
    pca = dp.PCA(n_components=4)
    pca.fit_transform(x)
    newX = pca.fit_transform(x)
    print(newX)
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(newX)):
        if y[i] == 0:
            red_x.append(newX[i][0])
            red_y.append(newX[i][1])
        elif y[i] == 1:
            blue_x.append(newX[i][0])
            blue_y.append(newX[i][1])
        else:
            green_x.append(newX[i][0])
            green_y.append(newX[i][1])
    fig, ax = plt.subplots()
    ax.plot(2,3)
    plt.rcParams['font.sans-serif']=['SimHei']
    ax.set_title('鸢尾花分类数据降维结果')
    plt.scatter(red_x, red_y, c='r', marker='*')
    plt.scatter(blue_x, blue_y, c='b', marker='.')
    plt.scatter(green_x, green_y, c='g', marker='+')
    plt.show()

    return

img = cv2.imread('mydrawing.jpg')
img1 = Add_GaussianNoise(img,0.0,0.2)
img2 = Add_PepperSaltNoise(img,5000)

cv2.imshow('origin',img)
cv2.imshow('AddGaussianNoise',img1)
cv2.imshow('Add_PSNoise',img2)
Decomposion()
cv2.waitKey(0)
cv2.DestroyAllWindows()
