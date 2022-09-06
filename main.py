from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from src import metrics
from src.datasets import *

import tensorflow as tf
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.manifold import TSNE

import cv2
from src.DCPN import DCPN
import argparse
import os
# GPU设置
def GpuInit():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    return session
# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--dataset',default='bearing_30_2',choices=['GSZ5','bearing_30_2'])
    parser.add_argument('--n_clusters',default=5,type=int)
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument('--epochs',default=2,type=int)
    parser.add_argument('--cae_weights',
                         help = 'This is argument must be given')
    parser.add_argument('--save_dir',default='results2/bearing_30_2')

    args = parser.parse_args()

    #print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":

    args = parse_args()

    print(args)

    sess = GpuInit()

    x, y = load_bearing_30_2()

    dc = DCPN(input_shape=(1024,1),n_clusters =5,datasets='bearing_30_2',x = x,y= y,
            pretrained = args.cae_weights,
            session = sess,
            lamda = 0,
            alpha = 1)

    dc.visulization(args.save_dir + '/embedding_1.png', save_dir=args.save_dir)

    dc.pretrain(x, batch_size= args.batch_size,
                epochs = args.epochs,
                save_dir=args.save_dir)
    # 随机1000个。TSNE+GMM测试后
    dc.evaluate(flag_all=True)
    dc.visulization(args.save_dir + '/embedding_init.png',save_dir = args.save_dir)



    dc.refineTrain(x,
                   batch_size = args.batch_size,
                   epochs = 2,
                   save_dir = args.save_dir,
                   second = True
                   )
    #  经过triple loss后的acc
    dc.evaluate(flag_all= True)
    dc.visulization(args.save_dir + '/embedding_refine.png',save_dir = args.save_dir,flag = 1)


















