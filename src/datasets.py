import numpy as np
import os
import cv2
import random
from PIL import Image
import scipy.io as scio
import numpy as np
import imageio
import h5py


def load_CWRU(data_path = './data/CWRU'):

    with h5py.File('.\\data\\CWRU\\CWRU_5000.h5','r') as hf:
        x = hf.get('train_x')[:]
        y = hf.get('train_y')[:]
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_GSZ5(data_path = './data/GSZ5' ):

    with h5py.File('.\\data\\GSZ5\\\GSZ5.h5','r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_data(data_path = './data/CWRU',name = 'CWRU'):
    x , y = None, None
    if name == 'GSZ5':
        x, y = load_GSZ5()
    elif name == 'CWRU':
        x, y = load_CWRU(data_path)
    else:
        print("Not found dataet!")
        return

    print(name + ':',x.shape)
    return x,y



