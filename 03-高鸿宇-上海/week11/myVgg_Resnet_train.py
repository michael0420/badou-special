import getdata
import keras_models
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# 使用GPU时设置此项
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #有多个GPU时可以指定只使用第几号GPU


class Training():
    def __init__(self, batch_size, epochs, img_shape, loss, optimizer, metrics, load_model_weight=None) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = keras_models.load_model(model_name, img_shape, loss, optimizer, metrics, load_model_weight)
        print('model complied success!')
        print("number of trainabale parameters:", self.model.count_params())
        self.model.summary()
        
    def fit_model(self, train_iter, valid_iter, weight_save_path):
        checkpointer = ModelCheckpoint(
            filepath=weight_save_path+'/model_epoch{epoch:2d}_val_loss{val_loss:.3f}.hdf5', verbose=1,
            save_best_only=True)
        self.model.fit_generator(train_iter, steps_per_epoch=train_iter.x.shape[0]//self.batch_size, epochs=self.epochs,
                                 validation_data=valid_iter, validation_steps=valid_iter.x.shape[0]//self.batch_size,
                                 verbose=1, callbacks=[checkpointer])

if __name__ == "__main__":
    # file_path = "data/cifar-10-batches-py/"
    # weight_save_path = 'week11/model/Vgg/'
    file_path = "/content/drive/My Drive/data/cifar-10-batches-py/"
    weight_save_path = '/content/drive/My Drive/week11/model/Resnet/'
    batch_size, epochs = 2, 100
    img_shape = (112,112,3)
    # train_iter, valid_iter = getdata.get_cifar_data_iter(file_path, batch_size, img_shape, is_train=True)
    optimizer = SGD(learning_rate=0.001, decay=5e-6, momentum=0.9, nesterov=False)
    loss='categorical_crossentropy'
    metrics=['accuracy']
    model_name = 'resnet-50'
    train_pipline = Training(batch_size, epochs, img_shape, loss, optimizer, metrics)
    # train_pipline.fit_model(train_iter, valid_iter, weight_save_path)
    
