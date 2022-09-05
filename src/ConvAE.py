from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Activation, Lambda,Concatenate,Conv1D,LeakyReLU
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import *
import numpy as np
import tensorflow as tf

def CAE_1D(input_shape=(1024,1),filters=[64,64,64,64,256]):
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

def CAE_2D(input_shape=(84,84,3),filters=[6,16,32,128,256,64,20]):

    model = Sequential()
    model.add(Conv2D(filters[0], 3, strides=1, padding='same', activation='relu', name='conv1', input_shape=input_shape))  #输出维度，窗口长度，步长，填充

    model.add(Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=1, padding='same', activation='relu', name='conv3'))

    model.add(Conv2D(filters[3], 3, strides=2, padding='same', activation='relu', name='conv4'))

    model.add(Conv2D(filters[4], 3, strides=1, padding='same', activation='relu', name='conv5'))

    model.add(Conv2D(filters[5], 3, strides=2, padding='same', activation='relu', name='conv6'))

    model.add(Conv2D(filters[6], 3, strides=1, padding='valid', activation='relu', name='conv7'))

    model.add(Flatten())
    model.add(Dense(units=filters[7]))

    model.add(BatchNormalization(name='embedding'))

    model.add(Dense(units=filters[6] * int(10) * int(10), activation='relu'))

    model.add(Reshape((int(10), int(10), filters[6])))

    model.add(Conv2DTranspose(filters[5], 3, strides=1, padding='same', activation='relu', name='deconv7'))

    model.add(Conv2DTranspose(filters[4], 3, strides=2, padding='valid', activation='relu', name='deconv6'))  #padding='valid'

    model.add(Conv2DTranspose(filters[3], 3, strides=1, padding='same', activation='relu', name='deconv5'))  # strides=1

    model.add(Conv2DTranspose(filters[2], 3, strides=2, padding='same', activation='relu', name='deconv4'))  #strides=2, padding='valid'

    model.add(Conv2DTranspose(filters[1], 3, strides=1, padding='same', activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2'))  #padding='valid'

    model.add(Conv2DTranspose(input_shape[2], 3, strides=1, padding='same', name='deconv1'))
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



# if __name__ == "__main__":
#     from time import time

#     # setting the hyper parameters
#     import argparse
#     parser = argparse.ArgumentParser(description='train')
#     parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
#     parser.add_argument('--n_clusters', default=10, type=int)
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--epochs', default=200, type=int)
#     parser.add_argument('--save_dir', default='results/temp', type=str)
#     args = parser.parse_args()
#     print(args)

#     import os
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)

#     # load dataset
#     from datasets import load_mnist, load_usps
#     if args.dataset == 'mnist':
#         x, y = load_mnist()
#     elif args.dataset == 'usps':
#         x, y = load_usps('data/usps')

#     # define the model
#     model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
#     plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
#     model.summary()

#     # compile the model and callbacks
#     optimizer = 'adam'
#     model.compile(optimizer=optimizer, loss='mse')
#     from keras.callbacks import CSVLogger
#     csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

#     # begin training
#     t0 = time()
#     model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
#     print('Training time: ', time() - t0)
#     model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

#     # extract features
#     feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
#     features = feature_model.predict(x)
#     print('feature shape=', features.shape)

#     # use features for clustering
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=args.n_clusters)

#     features = np.reshape(features, newshape=(features.shape[0], -1))
#     pred = km.fit_predict(features)
#     from . import metrics
#     print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
