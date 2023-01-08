[1]

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

[2]
#测试测试集有没有加载进来
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]

from tensorflow.keras import models
from tensorflow.keras import layers

# 构建了一个空模型，定义模型中各个层之间是串联的
network = models.Sequential()
# 添加一个全连接层，512表示节点个数、激活函数为relu、输入曾的节点个数为28*28
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# 类别的个数就是输出节点的个数，激活函数为softmax
# 为什么输出节点之后还要加上激活函数，将输出映射到【0，1】区间，可看成概率。
network.add(layers.Dense(10, activation='softmax'))

# 损失函数为交叉熵损失函数
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
      
[4]               
#将输入由2维转换为1维
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

[5]

network.fit(train_images, train_labels, epochs=5, batch_size = 128)

[6]

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss) 
print('test_acc', test_acc)

[7]

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_imagesr = test_images.reshape((10000, 28*28))
#import cv2
#img = plt.imread("12.jpg")
#test_imagesr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#test_imagesr = test_imagesr.reshape((1, 28*28))
res = network.predict(test_imagesr)
#print (res)

for i in range(10):
    print(res[i])

# 打开测试集的前十张图片并进行推理
for i in range(10):
    digit = test_images[i + 1]
    for j in range(res[i + 1].shape[0]):
        if (res[i + 1][j] == 1):
            print("the number for the picture is : ", j)
            plt.subplot(2, 5, (i + 1)), plt.xticks([]), plt.yticks([]), plt.title(j), plt.imshow(digit)
            break

plt.show()


