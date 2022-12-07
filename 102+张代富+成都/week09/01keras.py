from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt


def get_data():
    (train, train_labels), (test, test_labels) = mnist.load_data()
    print(train.shape, train_labels.shape)
    print(test.shape, test_labels.shape)
    print(set(train_labels))
    return train, train_labels, test, test_labels


def show_data(data):
    plt.imshow(data, cmap='Greys')
    plt.imshow(data)
    plt.show()


if __name__ == '__main__':
    # get_data()
    show_data(get_data()[0][0])
