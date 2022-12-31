"""
Canny边缘检测算法
    1. 对图像进行灰度化
    2. 对图像进行高斯滤波：
    根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
    可以有效滤去理想图像中叠加的高频噪声。
    3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
    4 对梯度幅值进行非极大值抑制
    5 用双阈值算法检测和连接边缘
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

warnings.filterwarnings('ignore')

image_path = './data/lenna.png'
image = plt.imread(image_path)
# matplotlib读取数据时，如果是png格式图片则是0-1之间
if image_path[-4:] == '.png':
    image = image * 255

# 1. 对图像进行灰度化  平均值也是灰度化的一种
image = image.mean(axis=-1)

# 2. 对图像进行高斯滤波
sigma = 0.5
dim = int(np.round(6 * sigma + 1))
if dim % 2 == 0:
    dim += 1

# 高斯核
gaussian_filter = np.zeros([dim, dim])
tmp = [i - dim // 2 for i in range(dim)]
n1 = 1 / (2 * math.pi * sigma ** 2)
n2 = -1 / (2 * sigma ** 2)
for i in range(dim):
    for j in range(dim):
        gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))

gaussian_filter = gaussian_filter / gaussian_filter.sum()

image_gaussian = np.zeros(image.shape)
image_padding = np.pad(image, ((dim // 2, dim // 2), (dim // 2, dim // 2)), 'constant')
# 卷积或者滤波的过程
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image_gaussian[i, j] = np.sum(image_padding[i:i + dim, j:j + dim] * gaussian_filter)
plt.figure(1)
plt.imshow(image_gaussian.astype(np.uint8), cmap='gray')
plt.show()

# 3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
image_grad_x = np.zeros(image_gaussian.shape)
image_grad_y = np.zeros(image_gaussian.shape)
image_grad = np.zeros(image_gaussian.shape)
image_padding = np.pad(image_gaussian, ((1, 1), (1, 1)), 'constant')

# sobel卷积核x方向和y方向卷积
dim = 3
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image_grad_x[i, j] = np.sum(image_padding[i:i + dim, j:j + dim] * sobel_kernel_x)
        image_grad_y[i, j] = np.sum(image_padding[i:i + dim, j:j + dim] * sobel_kernel_y)
        image_grad[i, j] = np.sqrt(image_grad_x[i, j] ** 2 + image_grad[i, j] ** 2)

plt.figure(2)
plt.imshow(image_grad.astype(np.uint8), cmap='gray')
plt.show()

# 4.对梯度幅值进行非极大值抑制
image_grad[image_grad == 0] = 0.00000001
# 正切值
tan = image_grad_y / image_grad_x
image_nms = np.zeros(image_grad.shape)
for i in range(1, image_grad.shape[0] - 1):
    for j in range(1, image_grad.shape[1] - 1):
        flag = True
        near_region = image_grad[i - 1:i + 2, j - 1:j + 2]
        # 从y轴开始旋转，每次旋转45度，TODO 此处线性插值的逻辑还需要理解
        if tan[i, j] >= 1:
            bilinear1 = (near_region[0, 2] - near_region[0, 1]) / tan[i, j] + near_region[0, 1]
            bilinear2 = (near_region[2, 0] - near_region[2, 1]) / tan[i, j] + near_region[2, 1]
            if not (image_grad[i, j] > bilinear1 and image_grad[i, j] > bilinear2):
                flag = False
        elif tan[i, j] > 0:
            bilinear1 = (near_region[0, 2] - near_region[1, 2]) * tan[i, j] + near_region[1, 2]
            bilinear2 = (near_region[2, 0] - near_region[1, 0]) * tan[i, j] + near_region[1, 0]
            if not (image_grad[i, j] > bilinear1 and image_grad[i, j] > bilinear2):
                flag = False
        elif tan[i, j] < 0:
            bilinear1 = (near_region[1, 0] - near_region[0, 0]) * tan[i, j] + near_region[1, 0]
            bilinear2 = (near_region[1, 2] - near_region[2, 2]) * tan[i, j] + near_region[1, 2]
            if not (image_grad[i, j] > bilinear1 and image_grad[i, j] > bilinear2):
                flag = False
        elif tan[i, j] <= -1:  # 使用线性插值法判断抑制与否
            bilinear1 = (near_region[0, 1] - near_region[0, 0]) / tan[i, j] + near_region[0, 1]
            bilinear2 = (near_region[2, 1] - near_region[2, 2]) / tan[i, j] + near_region[2, 1]
            if not (image_grad[i, j] > bilinear1 and image_grad[i, j] > bilinear2):
                flag = False
        if flag:
            image_nms[i, j] = image_grad[i, j]
plt.figure(3)
plt.imshow(image_nms.astype(np.uint8), cmap='gray')
plt.show()

# 5 用双阈值算法检测和连接边缘
threshold1 = image_grad.mean() * 0.5
threshold2 = threshold1 * 3

shed = []
for i in range(1, image_nms.shape[0] - 1):
    for j in range(1, image_nms.shape[1] - 1):
        if image_nms[i, j] >= threshold2:
            image_nms[i, j] = 255
            shed.append([i, j])
        elif image_nms[i, j] <= threshold1:
            image_nms[i, j] = 0

while shed:
    i, j = shed.pop()
    near_region = image_nms[i - 1:i + 2, j - 1:j + 2]
    if (near_region[0, 0] < threshold2) and (near_region[0, 0] > threshold1):
        image_nms[i - 1, j - 1] = 255  # 这个像素点标记为边缘
        shed.append([i - 1, j - 1])  # 进栈
    if (near_region[0, 1] < threshold2) and (near_region[0, 1] > threshold1):
        image_nms[i - 1, j] = 255
        shed.append([i - 1, j])
    if (near_region[0, 2] < threshold2) and (near_region[0, 2] > threshold1):
        image_nms[i - 1, j + 1] = 255
        shed.append([i - 1, j + 1])
    if (near_region[1, 0] < threshold2) and (near_region[1, 0] > threshold1):
        image_nms[i, j - 1] = 255
        shed.append([i, j - 1])
    if (near_region[1, 2] < threshold2) and (near_region[1, 2] > threshold1):
        image_nms[i, j + 1] = 255
        shed.append([i, j + 1])
    if (near_region[2, 0] < threshold2) and (near_region[2, 0] > threshold1):
        image_nms[i + 1, j - 1] = 255
        shed.append([i + 1, j - 1])
    if (near_region[2, 1] < threshold2) and (near_region[2, 1] > threshold1):
        image_nms[i + 1, j] = 255
        shed.append([i + 1, j])
    if (near_region[2, 2] < threshold2) and (near_region[2, 2] > threshold1):
        image_nms[i + 1, j + 1] = 255
        shed.append([i + 1, j + 1])

for i in range(image_nms.shape[0]):
    for j in range(image_nms.shape[1]):
        # if image_nms[i, j] not in [0, 255]:
        if image_nms[i, j] != 0 and image_nms[i, j] != 255:
            image_nms[i, j] = 0

plt.figure(4)
plt.imshow(image_nms.astype(np.uint8), cmap='gray')
plt.show()
