from keras.datasets import mnist
import numpy as np
from keras import models, layers
from keras.utils import to_categorical

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
X = train_features.reshape(-1, train_features.shape[1]*train_features.shape[2])
X = X.astype(np.float32) / 255
test_X = test_features.reshape(-1, test_features.shape[1]*test_features.shape[2])
test_X = test_X.astype(np.float32) / 255
y = to_categorical(train_labels)
test_y = to_categorical(test_labels)
net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
net.add(layers.Dense(10, activation='softmax'))
net.summary()

net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
net.fit(X, y, epochs=5, batch_size=128)
test_loss, test_acc = net.evaluate(test_X, test_y, verbose=1)
res = net.predict(test_X)
y_hat = np.argmax(res, axis=1)
print(y_hat)
