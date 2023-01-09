import getdata
import keras_models
import numpy as np
import pandas as pd

class Predict():
    def __init__(self, weight_path) -> None:
        self.model = keras_models.load_model(model_name, img_shape, None, None, None, weight_path)
        print('model compiled!\n')
    
    def predict_classes(self, test_iter):
        preds = self.model.predict(test_iter[0], verbose=1)
        preds = np.argmax(preds, axis=-1)
        labels = test_iter[1]
        true_nums = (preds == labels).sum()
        acc = true_nums / len(labels)
        print(f'acc:{acc}')
        return preds

if __name__ == "__main__":
    # file_path = "/content/drive/My Drive/data/cifar-10-batches-py/"
    # weight_path = '/content/drive/My Drive/week11/model/Vgg/'
    
    file_path = "data/cifar-10-batches-py/"
    weight_path = 'week11/model/Vgg/model_epoch16_val_loss1.222.hdf5'
    
    test_iter = getdata.get_cifar_data_iter(file_path, is_train=False)
    img_shape = (64,64,3)
    model_name = 'vgg'
    prediction_pipline = Predict(weight_path)
    preds = prediction_pipline.predict_classes(test_iter)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels = [classes[int(i)] for i in preds]
    df = pd.DataFrame({'labels': labels})
    print(df)