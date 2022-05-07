# -*- coding: utf-8 -*-
import os
import mxnet as mx
import shutil
import requests
import time
import cv2
import random
import numpy as np
import cPickle as pickle
from scipy.spatial.distance import cdist
from pathos.multiprocessing import Pool


import time


cv2.setNumThreads(0)
#p = Pool(32)


SIFT_ROOT = '/home/wangya/wine/distributed_4/data/files_2/pysift_train/'
sift = cv2.xfeatures2d.SIFT_create()

def load_sift():
    sift_files = os.listdir(SIFT_ROOT)[:]
    sifts = []
    for i in range(len(sift_files)):
        with open(os.path.join(SIFT_ROOT, sift_files[i])) as f:
            sd = pickle.load(f)
        sd[0] = np.array(sd[0])
        sd[1] = np.array(sd[1])
        sifts.append(sd)
    return np.array(sifts)


app_train_sifts = load_sift()
with open('./100_train_sifts.pkl', 'wb') as point:
    pickle.dump(app_train_sifts,point)
  