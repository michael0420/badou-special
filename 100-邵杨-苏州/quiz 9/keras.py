import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_image = test_images[510]

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(225, activation='relu', kernel_initializer='normal', input_shape=(28*28, )))
model.add(layers.Dense(10, activation='softmax', kernel_initializer='normal'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=500, verbose=2)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_loss)
print('test_acc: ', test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[510]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = model.predict(test_images)
print(res[510])
for i in range(res[510].shape[0]):
    if (res[510][i] == 1):
        print("the number for the picture is : ", i)
        break