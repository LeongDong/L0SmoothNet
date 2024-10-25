import numpy as np
import random
def hwc_to_chw(img):
    if img.ndim == 2:
        img = np.expand_dims(img,axis=0)
    return img

def batch(iterable,batch_size):
    b=[]
    for i,t in enumerate(iterable):
        b.append(t)
        if (i+1)%batch_size==0:
            yield b
            b=[]
    if len(b)>1:
        yield b

def split_dataset(dataset):
    dataset=list(dataset)
    random.shuffle(dataset)
    return {'train':dataset[:]}

def img_Normalize(x):
    return x/255.0

def mask_Normalize(x):
    return x/64

def fore_Mask(x):
    mask = np.zeros_like(x)
    H,W = mask.shape
    for i in range(H):
        for j in range(W):
            if(x[i,j] > 10):
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    return mask