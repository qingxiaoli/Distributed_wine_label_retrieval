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
import cnn_label_yd5 as cnn_label
import read_result
import time


cv2.setNumThreads(0)
#p = Pool(32)

N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed/data/files_2/pysift_train/'
sift = cv2.xfeatures2d.SIFT_create()

def load_sift():
    sift_files = os.listdir(SIFT_ROOT)[:N]
    sifts = []
    for i in range(len(sift_files)):
        with open(os.path.join(SIFT_ROOT, sift_files[i])) as f:
            sd = pickle.load(f)
        sd[0] = np.array(sd[0])
        sd[1] = np.array(sd[1])
        sifts.append(sd)
    return np.array(sifts)

def get_score(ss, des_te, kp_te, t0, t1, t2):
    def hat(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    kp_tr = ss[0]
    des_tr = ss[1]
    try:
        d = cdist(des_tr, des_te, 'euclidean')
    except:
        print (des_tr.shape, des_te.shape)
        return 0
	idx = np.argsort(d, axis=1)[:,:2]
	good = []
	if idx.shape[1] > 1:
		for i in range(idx.shape[0]):
			if d[i,idx[i,0]] * t0 < d[i,idx[i,1]]:
				good.append([i,idx[i,0]])
	else:      #当只有一个SIFT算子时
		for i in range(idx.shape[0]):
			good.append([i,idx[i,0]])
    good = np.array(good)
    if good.shape[0] > 5:
        X1 = np.ones([good.shape[0],3])
        X2 = np.ones([good.shape[0],3])
        X1[:,:2] = kp_tr[good[:,0],:]
        X2[:,:2] = kp_te[good[:,1],:]
        X1 = X1.transpose()
        X2 = X2.transpose()
        score = 0
        for i in range(t2):
            select = np.random.choice(good.shape[0], 4, replace=False)
            a = np.zeros([12,9])
            for i in range(4):
                idx = select[i]
                a[i*3:i*3+3,:] = np.kron(X1[:,idx], hat(X2[:,idx]))
            _, _, V = np.linalg.svd(a)
            H = np.reshape(V[8,:], [3,3]).transpose()
            X2_ = np.dot(H, X1)
            if np.min(np.abs(X2_[2,:])) < 1e-5:
                continue
            du = X2_[0,:] / X2_[2,:] - X2[0,:] / X2[2,:]
            dv = X2_[1,:] / X2_[2,:] - X2[1,:] / X2[2,:]
            s = du*du + dv*dv
            score = max(score, np.sum(s<t1**2))
        return score
    else:
        return 0

def get_score2(ss, des_te, kp_te, t0, t1, t2):
    def hat(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    kp_tr = ss[0]
    des_tr = ss[1]
    try:
        d = cdist(des_tr, des_te, 'euclidean')
    except:
        print (des_tr.shape, des_te.shape)
        return np.zeros(kp_te.shape[0])

    idx = np.argsort(d, axis=1)[:,:2]
    good = []
    for i in range(idx.shape[0]):
        if d[i,idx[i,0]] * t0 < d[i,idx[i,1]]:
            good.append([i,idx[i,0]])
    good = np.array(good)
    if good.shape[0] > 5:
        X1 = np.ones([good.shape[0],3])
        X2 = np.ones([good.shape[0],3])
        X1[:,:2] = kp_tr[good[:,0],:]
        X2[:,:2] = kp_te[good[:,1],:]
        X1 = X1.transpose()
        X2 = X2.transpose()
        score = -1 
        mask = np.zeros(good.shape[0])
        for i in range(t2):
            select = np.random.choice(good.shape[0], 4, replace=False)
            a = np.zeros([12,9])
            for i in range(4):
                idx = select[i]
                a[i*3:i*3+3,:] = np.kron(X1[:,idx], hat(X2[:,idx]))
            _, _, V = np.linalg.svd(a)
            H = np.reshape(V[8,:], [3,3]).transpose()
            X2_ = np.dot(H, X1)
            if np.min(np.abs(X2_[2,:])) < 1e-5:
                continue
            du = X2_[0,:] / X2_[2,:] - X2[0,:] / X2[2,:]
            dv = X2_[1,:] / X2_[2,:] - X2[1,:] / X2[2,:]
            s = du*du + dv*dv
            tmp = s < (t1**2)
            if np.sum(tmp) > score:
                score = np.sum(tmp)
                mask = tmp
        matched = np.zeros(kp_te.shape[0])
        for m, boo in zip(good, mask):
            if boo:
                matched[m[1]] = 1
        return matched
    else:
        return np.zeros(kp_te.shape[0])

def crop_img(im, mask):
    x0, y0 = 0, 0
    x1, y1 = mask.shape
    x1 -= 1
    y1 -= 1
    while np.sum(mask[x0,:])==0:
        x0 += 1
    while np.sum(mask[x1,:])==0:
        x1 -= 1
    while np.sum(mask[:,y0])==0:
        y0 += 1
    while np.sum(mask[:,y1])==0:
        y1 -= 1
    im = im[x0:x1, y0:y1]
    im = cv2.resize(im, (128, 192))
    return im


# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed/data/files_2/app_train_sifts.pkl', 'rb') as point:
    app_train_sifts = pickle.load(point) 
app_train_names = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed/data/files_2/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand = {}
    for item in brand_temp:
        brand[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names:
    app_brand.append(brand[int(k)])
app_brand_dic = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic:
        app_brand_dic[app_brand[i]] = [i]
    else:
        app_brand_dic[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed/2/model/myresnext_wd0.00008',3)
#s = sym.get_internals()['fullyconnected0_output']
model = mx.module.Module(symbol=sym, context=mx.gpu())
model.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model.set_params(arg_params=arg_params, aux_params=aux_params)


def match(test_img, RANGE, THRES0, THRES1, T2):
    global app_train_sifts, app_train_names, brand, app_brand_dic, model

    start_time = time.time()

    test_image_name = os.path.split(test_img)[-1].split('.')[0]
    test_img = '/home/wangya/wine/distributed/data/files_2/test/'+test_image_name+'.jpg'
    im = cv2.imread(test_img)
  
    mask_file = '/home/wangya/wine/distributed/data/files_2/test_mask/'+test_image_name+'.jpg'
    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE) #opencv imread gray image
    
    factor = 500. / max(im.shape)
    im = cv2.resize(im, (int(im.shape[1]*factor), int(im.shape[0]*factor)))

    croped_dir = '/home/wangya/wine/distributed/data/files_2/test_crop/'+test_image_name+'.jpg'
    brand=cnn_label.main(croped_dir, model)

    pos_all, des_all = sift.detectAndCompute(im, mask)
    order = np.argsort([-k.response for k in pos_all]) 
    pos, des = [], []
    for i in range(min(500, order.shape[0])):
        pos.append(pos_all[order[i]].pt)
        des.append(des_all[order[i]])
    des = np.array(des, dtype=np.uint8)
    pos = np.array(pos)
    zero = np.zeros(N, np.float32)

    
    # candi = app_train_sifts[[i for i in range(len(app_brand)) if app_brand[i] in brand]]
    candi = app_train_sifts[reduce(lambda x, y: x+y, [app_brand_dic[bb] for bb in brand])]
    n_candi = len(candi)
    other_score = [get_score2(ss, des, pos, THRES0, THRES1, T2) for ss in candi]
    other_score = np.array(other_score)

    des_weight = other_score.sum(0)
    des_weight[des_weight==0] = 1      
    des_weight = n_candi / des_weight
    npscore = np.inner(other_score, des_weight)
    best = np.argsort(npscore)[-1:-6:-1]
    end_time = time.time() 
    ti = end_time-start_time
   # print (time.clock()-t)
    result_imgs = []
    # app_train_names_sub = app_train_names[[ii for ii in range(len(app_brand)) if app_brand[ii] in brand]]
    app_train_names_sub = app_train_names[reduce(lambda x, y: x+y, [app_brand_dic[bb] for bb in brand])]
    for i in best:
        result_img = app_train_names_sub[i]
        result_imgs.append(result_img)
    # print(app_brand[app_brand == brand][best])
    return result_imgs,ti


if __name__ == '__main__':
   # test_img = './1.jpg'
    RANGE =2000
    THRES0 = 1.5
    THRES1 =5.0
    T2 = 100
    t = 0 #表示时间
    dir = '/home/wangya/wine/distributed/data/files_2/test'
    for img in os.listdir(dir)[20000:40000]:
        print(img)
        test_img = os.path.join(dir,img)
        result_imgs,ti= match(test_img,RANGE,THRES0,THRES1,T2)
        t =t+ti
        for result_img in result_imgs:
            result_img = result_img + '.jpg'
            result_img = os.path.join('/home/wangya/wine/distributed/data/files_2/train',result_img)
            read_result.cp_img(test_img,result_img,'/home/wangya/wine/distributed/2/result_5brand_5')
    print('total time is :',t)   
    print('avg time is :',t/10000)
    