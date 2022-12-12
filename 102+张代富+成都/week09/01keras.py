from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.utils import to_categorical


def get_data():
    (_train, _train_labels), (_test, _test_labels) = mnist.load_data()
    return _train, _train_labels, _test, _test_labels


def deal_data():
    _train, _train_labels, _test, _test_labels = get_data()
    _train = _train.reshape((_train.shape[0], -1))
    _train = _train.astype('float32') / 255
    _test = _test.reshape((_test.shape[0], 28 * 28))
    _test = _test.astype('float32') / 255
    # one-hot编码
    _train_labels = to_categorical(_train_labels)
    _test_labels = to_categorical(_test_labels)
    return _train, _train_labels, _test, _test_labels


def build_model():
    """
    创建模型
    参考： https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Sequential
    """
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return network


if __name__ == '__main__':
    train, train_labels, test, test_labels = deal_data()
    net = build_model()
    net.fit(train, train_labels, epochs=5, batch_size=128)
    loss, accuracy = net.evaluate(test, test_labels)
    print(f"accuracy: {accuracy}")
