from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers import concatenate, add

class My_Vgg(object):
    def __init__(self, img_shape, loss, optimizer, metrics, model_load_weights=None) -> None:
        self.img_shape = img_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model_load_weights = model_load_weights
        self.model = self.compile_vgg()
    
    def compile_vgg(self):
        input_img = Input(shape=self.img_shape)
        output = self.vgg(input_img, channels=[64,128,256,512])
        model = Model(inputs=input_img, outputs=output)
        
        if self.model_load_weights is not None:
            model.compile()
            model.load_weights(self.model_load_weights)
        else:
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        
        return model
    
    def vgg(self, inputs, channels):
        # vgg16 结构：2个2层conv的vgg块+3个3层conv的vgg块+3层全连接+softmax激活
        output_blk1 = self.vgg_block(inputs, channels[0], layers=2)
        output_blk2 = self.vgg_block(output_blk1, channels[1], layers=2)
        output_blk3 = self.vgg_block(output_blk2, channels[2], layers=3)
        output_blk4 = self.vgg_block(output_blk3, channels[3], layers=3)
        output_blk5 = self.vgg_block(output_blk4, channels[3], layers=3)
        output_fc = self.cls_block(output_blk5)
        output = Activation('softmax')(output_fc)
        return output
    
    def vgg_block(self, x, channel, layers):
        for _ in range(layers):
            x = Conv2D(filters=channel, kernel_size=3, padding='same')(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        return x
    
    def cls_block(self, x):
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(10, activation=None)(x)
        return x
    
class My_Resnet50(object):
    def __init__(self, img_shape, loss, optimizer, metrics, model_load_weights=None) -> None:
        self.img_shape = img_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model_load_weights = model_load_weights
        self.model = self.compile_resnet50()
    
    def compile_resnet50(self):
        input_img = Input(shape=self.img_shape)
        output = self.resnet(input_img, channels=[64,128,256,512])
        model = Model(inputs=input_img, outputs=output)
        
        if self.model_load_weights is not None:
            model.compile()
            model.load_weights(self.model_load_weights)
        else:
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        
        return model
    
    def resnet(self, inputs, channels):
        # 定义resnet50结构 headblock(conv,bn,relu,maxpooling)+3*resblock+4*resblock+6*resblock+3*resblock+clsblock
        output_headnlk = self.head_block(inputs, channels[0])
        output_blk1 = self.res_block(output_headnlk, channels[0], num_blks=3)
        output_blk2 = self.res_block(output_blk1, channels[1], num_blks=4)
        output_blk3 = self.res_block(output_blk2, channels[2], num_blks=6)
        output_blk4 = self.res_block(output_blk3, channels[3], num_blks=3)
        output_fc = self.cls_block(output_blk4)
        output = Activation('softmax')(output_fc)
        return output
    
    def head_block(self, x, channel):
        x = Conv2D(filters=channel, kernel_size=7, strides=(2,2), padding='same')(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        return x
    
    def res_block(self, x, channel, num_blks):
        for i in range(num_blks):
            if i == 0:
                x = self.make_res_layers(x, channel, is_first_layer=True)
            else:
                x = self.make_res_layers(x, channel, is_first_layer=False)
        return x
    
    def make_res_layers(self, x, channel, is_first_layer=True):
        y = Conv2D(filters=channel, kernel_size=1, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=channel, kernel_size=3, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=channel*4, kernel_size=1, padding='same')(y)
        y = BatchNormalization()(y)
        if is_first_layer:
            x = Conv2D(filters=channel*4, kernel_size=1, padding='same')(x)
            x = BatchNormalization()(x)
        y = add([x,y])
        y = Activation('relu')(y)
        return y
    
    def cls_block(self, x):
        x = AveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(10, activation=None)(x)
        return x

def load_model(model_name, img_shape, loss, optimizer, metrics, model_load_weights=None):
    if model_name == 'vgg':
        model = My_Vgg(img_shape, loss, optimizer, metrics, model_load_weights).model
    if model_name == 'resnet-50':
        model = My_Resnet50(img_shape, loss, optimizer, metrics, model_load_weights).model
    return model