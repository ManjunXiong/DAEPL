import numpy as np
import os
import cv2
import random
from PIL import Image

import imageio
'''加载数据
'''

def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def load_GSZ5(data_path = './data/GSZ5' ):

    import h5py
    import numpy as np

    with h5py.File('.\\data\\GSZ5\\GSZ5.h5','r') as hf:
        x = hf.get('data')[:900]
        y = hf.get('labels')[:900]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_bearing_30_2(data_path = './data/bearing_30_2' ):

    import h5py
    import numpy as np

    with h5py.File('.\\data\\bearing_30_2\\bearing_30_2.h5','r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_Gearset_30_2(data_path = './data/Gearset_30_2'):

    import h5py
    import numpy as np

    with h5py.File(".\\data\\Gearset_30_2\\Gearset_30_2.h5",'r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_MIX(data_path = './data/MIX'):

    import h5py
    import numpy as np

    with h5py.File(".\\data\\MIX\\MIX.h5",'r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_data(data_path = '.\\data\\bearing_30_2',name = 'bearing_30_2'):
    x , y = None, None
    if name == 'GSZ5':
        x, y = load_GSZ5()
        x = x / 255
    elif name == 'bearing_30_2':
        x, y = load_bearing_30_2(data_path)
        x = x / 255
    elif name == 'Gearset_30_2':
        x, y = load_Gearset_30_2(data_path)
        x = x / 255
    elif name == 'MIX':
        x, y = load_MIX(data_path)
        x = x / 255
    else:
        print("Not found dataet!")
        return

    print(name + ':',x.shape)
    return x,y



