import numpy as np
import cv2
import pickle
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def unpickle(file_path, file):
    with open(file_path+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar_data_iter(file_path, img_shape, batch_size = 128, is_train=True):
    if is_train:
        # 读取训练集数据
        cifar_label = []
        cifar_data = []
        for i in range(1, 6):
            data = unpickle(file_path, f'data_batch_{i}')
            cifar_label += list(data[b'labels'])
            cifar_data += list(data[b'data'])
    else:
        # 读取训练集数据
        data = unpickle(file_path, f'test_batch')
        cifar_label = list(data[b'labels'])
        cifar_data = list(data[b'data'])
        
    # 读入的格式为n*(3*32*32)因此先将数据恢复成n*3*32*32
    labels=np.array(cifar_label).astype(np.uint8)
    cifar_data=np.array(cifar_data).astype(np.float32).reshape(-1,3,32,32).transpose(0,2,3,1)
    # 原图为32*32因此将图像size成128*128传入网络
    features = np.array([cv2.resize(each, img_shape) for each in cifar_data])
    if is_train:
        # 每个像素值/255（像素归一化），同时将20%的数据划分为验证集
        train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2, horizontal_flip=True)
        # 对label数据进行one-hot编码
        labels = to_categorical(labels)
        train_iter = train_datagen.flow(x=features, y=labels, batch_size=batch_size, shuffle= True, subset='training')
        valid_iter = train_datagen.flow(x=features, y=labels, batch_size=batch_size, shuffle= True, subset='validation')
        return train_iter, valid_iter
    else:
        features = features / 255
        return (features, labels)