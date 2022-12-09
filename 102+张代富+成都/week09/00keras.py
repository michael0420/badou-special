from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt


def get_data():
    (train, train_labels), (test, test_labels) = mnist.load_data()
    return train, train_labels, test, test_labels


def show_data(data, c_map=None):
    plt.imshow(data, cmap=c_map)
    plt.show()


if __name__ == '__main__':
    image = get_data()[0][0]
    show_data(image)
    show_data(image, c_map='gray')
