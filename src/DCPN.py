from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from . import metrics

from .ConvAE import CAE_2D,CAE_1D
from sklearn.mixture import GaussianMixture

import tensorflow as tf
import matplotlib
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn.neighbors import kneighbors_graph

from itertools import cycle, islice
from sklearn import cluster, datasets, mixture,manifold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
import cv2

from IPython import embed
import tensorflow as tf
import umap
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn import manifold
matplotlib.use('Agg')


class DCPN(object):
    def __init__(self,
                 input_shape=(84,84,3),
                 n_clusters=5,
                 datasets = 'cwru2D',
                 x = None,
                 y = None,
                 pretrained = None,
                 session = None,
                 lamda = 0.1,
                 alpha = 100):

        super(DCPN,self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape

        self.pretrained = pretrained

        self.datasets = datasets
        self.x = x
        self.y = y

        self.lamda = lamda
        self.alpha = alpha

        # Sample loss
        self.gamma_tr = 1.0
        self.margin = 0.2

        self.sess = session

        # self.cae, self.encoder,self.exlayer1, self.exlayer2,self.exlayer3,self.exlayer4= self.get_model(self.datasets) #  AE model
        self.cae, self.encoder = self.get_model(
            self.datasets)
        self.gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter = 200000)  #  GMM


        self.learning_rate = tf.Variable(0.000001, trainable=False, name='learning_rate')#0.000001
        #self.learning_rate = tf.Variable(self.base_lr * np.power(1 + self.gamma_lr * self.iter_cnn, - self.power_lr),
        #                                 trainable=False, name="learning_rate")


        if self.pretrained is not None:
            print("Load pretrained model...")
            self.load_weights(self.pretrained)
            print("Model %s load ok"%self.pretrained)
        else:
            self.sess.run(tf.global_variables_initializer())

    #   epochs设置。compile、fit，AE训练，没有特殊的loss
    def pretrain(self, x, batch_size=256, epochs=100, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger  #   记录迭代过程
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])  #   训练
        print('Pretraining time: ', time() - t0)
        self.cae.save_weights(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = save_dir + '/pretrain_cae_model.h5'

    def refineTrain(self, x, batch_size=256, epochs=100, save_dir='results/temp', second=True):

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.cluster(x)

        self.gmm.fit(self.z_reduced)
        y_pred = self.gmm.predict(self.z_reduced)

        # self.dbscan = cluster.DBSCAN(eps=2)
        # y_pred = self. dbscan.fit_predict(self.z_reduced)

        self.z_label = y_pred

        # acc = metrics.acc(self.y, y_pred)
        # nmi = metrics.nmi(self.y, y_pred)
        # ari = metrics.ari(self.y, y_pred)
        # print("after tsne and gmm without triple loss...")
        # print('acc = %.4f, nmi = %.4f, ari = %.4f' % (acc, nmi, ari))

        print("Refine Network...")
        self.proba = self.gmm.predict_proba(self.z_reduced)
        self.p = self.proba

        # restruct loss
        self.input_x = tf.placeholder(tf.float32,
                                      shape=[None, self.input_shape[0], self.input_shape[1]])  #   输入的形状、格式

        restruct = self.cae(self.input_x)

        self.loss_restruct = K.mean(K.square(restruct - self.input_x))  # x'和x的差异

        # dis_loss
        self.idx_1 = tf.placeholder(tf.int32, shape=[None])
        self.idx_2 = tf.placeholder(tf.int32, shape=[None])
        self.idx_3 = tf.placeholder(tf.int32, shape=[None])

        self.propos = tf.placeholder(tf.float32, shape=[None])

        z = self.encoder(self.input_x)

        z_anc = tf.gather(z, self.idx_1)
        z_pos = tf.gather(z, self.idx_2)
        z_neg = tf.gather(z, self.idx_3)

        d_pos = tf.reduce_sum(tf.square(z_anc - z_pos), -1)
        d_neg = tf.reduce_sum(tf.square(z_anc - z_neg), -1)

        d_pos = tf.multiply(d_pos, self.propos)

        self.loss_dis = tf.reduce_mean(tf.maximum(0., self.margin + self.gamma_tr * d_pos - d_neg))

        # all_loss
        self.loss = self.loss_restruct + self.loss_dis * self.alpha

        self.train_step = self.optimizer.minimize(self.loss)

        # initial
        if second == False:
            self.sess.run(tf.global_variables_initializer())

        if self.pretrained is not None:
            self.cae.load_weights(self.pretrained)

        # train
        sz = x.shape[0] // batch_size - 1

        for epoch in range(epochs):

            index_rand = np.random.permutation(x.shape[0])

            all_loss = 0
            for i in range(sz):
                tx = x[index_rand[i * batch_size:(i + 1) * batch_size]]

                idx_1, idx_2, idx_3, propos = self.getSample(
                    self.z_label[index_rand[i * batch_size:(i + 1) * batch_size]],
                    self.p[index_rand[i * batch_size:(i + 1) * batch_size]])


                _, loss, loss_re, loss_dis = self.sess.run([self.train_step, self.loss, self.loss_restruct,
                                                            self.loss_dis],
                                                           feed_dict={self.input_x: tx,
                                                                      self.idx_1: idx_1,
                                                                      self.idx_2: idx_2,
                                                                      self.idx_3: idx_3,
                                                                      self.propos: propos,
                                                                      })

                all_loss += loss

            print('Epoch:%4d Loss: %.4f %.4f re: %.4f dis: %.4f' % (epoch, all_loss / sz, loss, loss_re, loss_dis))

        self.cae.save_weights(save_dir + '/refine_cae_model.h5')

        return


    def cluster(self, x, reduce = True):

        z = self.encoder.predict(x)

        if reduce :
            # z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
            z_reduced = umap.UMAP(n_components=2,random_state=42).fit_transform(z)
            #z_reduced = PCA(n_components=2).fit_transform(z)

            #z_reduced = FactorAnalysis(n_components=2).fit_transform(z)
           # z_reduced = FastICA(n_components=3, random_state=12).fit_transform(z)
           # z_reduced = manifold.Isomap(n_neighbors=3, n_components=2, n_jobs=-1).fit_transform(z)

        else:
            z_reduced = z

        self.gmm.fit(z_reduced)
        z_label = self.gmm.predict(z_reduced)

        self.z_reduced = z_reduced
        self.z_label = z_label

        return z_label

    def evaluate(self,flag_all = False):

        index = np.random.randint(self.x.shape[0],size = 500)

        x = self.x[index]
        y = self.y[index]

        if flag_all:
            x = self.x
            y = self.y

        y_pred = self.cluster(x, False)

        acc = metrics.acc(y, y_pred)
        nmi = metrics.nmi(y, y_pred)
        ari = metrics.ari(y, y_pred)

        print('Clustering without reduction: acc = %.4f, nmi = %.4f, ari = %.4f' % (acc, nmi, ari))

        y_pred = self.cluster(self.x)


        acc = metrics.acc(y, y_pred)
        nmi = metrics.nmi(y, y_pred)
        ari = metrics.ari(y, y_pred)

        print('Clustering after reduction: acc = %.4f, nmi = %.4f, ari = %.4f' % (acc,nmi,ari))

        return acc,nmi,ari

    def load_weights(self,weights_path):
        self.cae.load_weights(weights_path)

    def getSample(self,label,prob):
        st,num = np.unique(label,return_counts=True)
        num_triplet = 0

        for i in range(st.shape[0]):
            num_triplet = num_triplet + (num[i] + 1) * num[i] // 2

        anc = np.zeros(num_triplet, np.int32)
        pos = np.zeros(num_triplet, np.int32)
        neg = np.zeros(num_triplet, np.int32)

        propos = np.ones(num_triplet,np.float32)

        index = 0
        for i in range(st.shape[0]):

            temp = np.argwhere(label == st[i])
            netemp = np.argwhere(label != st[i])

            for t in range(temp.shape[0]):
                for k in range(t+1,temp.shape[0]):
                    anc[index] = temp[t]
                    pos[index] = temp[k]

                    if prob[temp[t]].max() < 0.7:
                        break
                    elif prob[temp[k]].max() < 0.7:
                        continue

                    propos[index] = propos[index] - np.abs(prob[temp[t]].max() - prob[temp[k]].max())

                    c = np.random.randint(netemp.shape[0])
                    neg[index] = netemp[c]

                    index += 1

        return anc,pos,neg,propos

    def getPQ(self,q):
        p = q ** 2 / np.sum(q,axis=0)
        p = (p.T / np.sum(p,axis=1)).T
        return q, p

    def visulization(self,name = 'temp.png',save_dir = None,flag=0):


        x = self.x
        y = self.y

        z = self.encoder.predict(x)

        print('Embedding shape:',z.shape)
        # z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
        z_reduced = umap.UMAP(n_components=2,random_state=42).fit_transform(z)
        if flag == 0:
            np.save(save_dir + '/x.npy',z)
            np.save(save_dir + '/z_reduce.npy',z_reduced)
            np.save(save_dir + '/z_label.npy',y)
        else:
            np.save(save_dir + '/rx.npy',z)
            np.save(save_dir + '/rz_reduce.npy',z_reduced)
            np.save(save_dir + '/rz_label.npy',y)


        y = y.astype(int)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00','#171717']),
                                      int(max(y) + 1))))

        colors = np.append(colors, ["#000000"])

        plt.scatter(z_reduced[:, 0], z_reduced[:, 1], s=10, color=colors[y],alpha=0.5)

        plt.axis('off')
        plt.savefig(name)
        plt.close('all')

    def get_model(self, datasets='GSZ5'):
        if 'GSZ5' in datasets:
            cae = CAE_1D(self.input_shape, filters=[3,8,16,32,64,128,32,128])
        elif 'Gearset_30_2' in datasets:
            cae = CAE_1D(self.input_shape, filters=[3, 8, 16, 32, 64, 128, 32, 128])
        elif 'bearing_30_2' in datasets:
            cae = CAE_1D(self.input_shape, filters=[3,8,16,32,64,128,32,128])

        embedding = cae.get_layer(name='embedding').output
        encoder = Model(inputs=cae.input, outputs=embedding)
        # exlayer1 = Model(inputs = cae.input, outputs=cae.layers[2].output)
        # exlayer2 = Model(inputs=cae.input, outputs=cae.layers[5].output)
        # exlayer3 = Model(inputs=cae.input, outputs=cae.layers[8].output)
        # exlayer4 = Model(inputs=cae.input, outputs=cae.layers[11].output)

        # return cae, encoder,exlayer1,exlayer2,exlayer3,exlayer4
        return cae, encoder



