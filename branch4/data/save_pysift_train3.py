import os
import cPickle as pickle
import numpy as np
import cv2
import requests


def process_img(im, name):
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    factor = 500. / max(im.shape)    # this mean use pre 500 sifts?
    im = cv2.resize(im, (int(im.shape[1]*factor), int(im.shape[0]*factor)))
    mask = cv2.imread('/home/wangya/wine/distributed_4/data/files_3/train_mask/%s'%name.replace('.jpg','.png'), 0)
    sift = cv2.xfeatures2d.SIFT_create()
    pos_all, des_all = sift.detectAndCompute(im, mask) #return keypoints and describer
    order = np.argsort([-k.response for k in pos_all]) 
    pos, des = [], []
    for i in range(min(500, order.shape[0])):   #acrroding response to sort the keypoints and describer
        pos.append(pos_all[order[i]])
        des.append(des_all[order[i]])
    return pos, des     


def do(x, n):
    train_path = '/home/wangya/wine/distributed_4/data/files_3/train'
    train_pics = os.listdir(train_path)
    train_save_path = '/home/wangya/wine/distributed_4/data/files_3/pysift_train'
    for im in train_pics[x:x+n]:
        name = im.replace('.jpg', '.pkl')
        if os.path.exists('/home/wangya/wine/distributed_4/data/files_3/pysift_train/'+name) or not im.endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(train_path, im))
        pos, des = process_img(img, im)  # get sorted sift describer
        with open(os.path.join(train_save_path, name), 'w') as f:
            pickle.dump([map(lambda x:np.array(x.pt), pos), np.array(des).astype(np.uint8)], f)
