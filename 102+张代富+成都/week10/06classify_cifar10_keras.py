import os
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_classes = 10


class CnnNet:
    def __init__(self):
        (self.train, self.train_label), (self.test, self.test_label) = cifar10.load_data()
        self.train = self.train[:1000]
        self.train_label = self.train_label[:1000]
        self.test = self.test[:1000]
        self.test_label = self.test_label[:1000]
        self.train = self.train.reshape((-1, 32, 32, 3)) / 255
        self.test = self.test.reshape((-1, 32, 32, 3)) / 255

        self.model = Sequential([
            Conv2D(32, 5, 1, 'same', activation=tf.nn.relu),
            MaxPooling2D(2, 2, 'same'),
            Conv2D(64, 3, 1, 'same', activation=tf.nn.relu),
            MaxPooling2D(2, 2, 'valid'),
            Flatten(),
            Dense(16 * 16, activation=tf.nn.relu),
            Dense(64, activation=tf.nn.relu),
            Dense(10, activation=tf.nn.softmax)
        ])

    def compile(self):
        self.model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

    def fit(self):
        check = ModelCheckpoint('./model/weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_loss', save_best_only=True)
        self.model.fit(self.train, self.train_label, epochs=10, batch_size=500, callbacks=[check])

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test, self.test_label)
        return loss, accuracy

    def predict(self):
        predictions = self.model.predict(self.test)
        return np.argmax(predictions, axis=1)


def main():
    cnn = CnnNet()
    cnn.compile()
    cnn.fit()
    print(f"测试集的损失和精确度分别为{cnn.evaluate()}")
    print(cnn.predict().shape)
    print(cnn.model.summary())


if __name__ == '__main__':
    main()
