from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Activation, Lambda,Concatenate,Conv1D,LeakyReLU
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import *
import numpy as np
import tensorflow as tf




#  卷积自编码网络
def CAE_1D(input_shape=(1024,1),filters=[64,64,64,64,256]):  #cwru效果好
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv2'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv3'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv4'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Flatten())

    model.add(Dense(units=256, activation='linear'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(units=16 * int(40), activation='linear', name='embedding'))
    model.add(Reshape((int(40), 16)))

    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='valid', activation='linear', name='deconv4'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='linear', name='deconv3'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='linear', name='deconv2'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(input_shape[1], kernel_size=3, strides=1, padding='valid', activation='linear', name='deconv1'))
    model.add(Conv1D(input_shape[1], kernel_size=3, strides=1, padding='same', activation='linear', name='deconv0'))
    model.summary()
    return model



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 10))
    return z_mean + K.exp(z_log_var / 2) * epsilon


class Gaussian(Layer):
    """这是个简单的层，只为定义q(z|y)中的均值参数，每个类别配一个均值。
    输出也只是把这些均值输出，为后面计算loss准备，本身没有任何运算。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, 10),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])



