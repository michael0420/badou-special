"""
@author:Rai
用keras实现神经网络
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras import  models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1.读取训练和测试数据
(train_data, train_label), (test_data, test_label) = mnist.load_data()
print('train data shape:', train_data.shape)
print('train label shape:', train_label.shape)
print('test data shape:', test_data.shape)
print('test label shape:', test_label.shape)

# 打印图片
plt.imshow(train_data[6], cmap=plt.cm.binary)
plt.show()
print('train_data[6]:', train_label[6])


# 2.定义神经网络
model = models.Sequential()
model.add(layers.Dense(units=256,
                       activation='relu',
                       input_shape=(28*28,)))
model.add(layers.Dense(units=10,
                       activation='softmax'))

# 定义优化器，目标函数，模型评估标准
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 3.输入数据处理
# 归一化
n, h, w = train_data.shape
train_x = train_data.reshape(n, h * w)
train_x = train_x.astype('float32') / 255

n, h, w = test_data.shape
test_x = test_data.reshape(n, h * w)
test_x = test_x.astype('float32') / 255

# 标签 -> one-hot
train_y = to_categorical(train_label)
test_y = to_categorical(test_label)

# 4.神经网络训练
print("---训练结果---")
model.fit(x=train_x, y=train_y, batch_size=256, epochs=5, verbose=2)
# verbose: set the prograss bar (0 = silent, 1 = progress bar, 2 = one line per epoch)

# 5.测试数据
print('\n---测试结果---')
test_loss, test_acc = model.evaluate(x=test_x, y=test_y, batch_size=128, verbose=2)
print('test loss:{:.4f}, test acc:{:.4f}'.format(test_loss, test_acc))

# 6.推理
img = test_data[6]
img = img.reshape(1, 28*28)
y_pred = model.predict(img)
y_pred_label = y_pred.argmax()
print('预测值的one hot:\n', y_pred)
print('预测值：', y_pred_label)
print('目标值:', test_label[6])
