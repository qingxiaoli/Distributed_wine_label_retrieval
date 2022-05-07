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
from joblib import Parallel, delayed


cv2.setNumThreads(0)
sift = cv2.xfeatures2d.SIFT_create()
#p = Pool(32)

####################################第一个任务
N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed/data/files_1/pysift_train/'

# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed/data/files_1/app_train_sifts.pkl', 'rb') as point:
    app_train_sifts1 = pickle.load(point)
# app_train_sifts1 = {} 
app_train_names1 = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed/data/files_1/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand1 = {}
    for item in brand_temp:
        brand1[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names1:
    app_brand.append(brand1[int(k)])
app_brand_dic1 = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic1:
        app_brand_dic1[app_brand[i]] = [i]
    else:
        app_brand_dic1[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed/1/model/myresnext_wd0.00015',17)
#s = sym.get_internals()['fullyconnected0_output']
model1 = mx.module.Module(symbol=sym, context=mx.gpu(0))
model1.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model1.set_params(arg_params=arg_params, aux_params=aux_params)


####################################第一个任务
N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed_4/data/files_1/pysift_train/'

# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed_4/data/model/part1/cnn_sift/100_train_sifts.pkl', 'rb') as point:
    app_train_sifts1 = pickle.load(point) 
# app_train_sifts1 = {}/home/wangya/wine/distributed_4/data/model/part1/cnn_sift
app_train_names1 = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed_4/data/files_1/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand1 = {}
    for item in brand_temp:
        brand1[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names1:
    app_brand.append(brand1[int(k)])
app_brand_dic1 = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic1:
        app_brand_dic1[app_brand[i]] = [i]
    else:
        app_brand_dic1[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed_4/data/model/part1/myresnext_wd0.00008',24)
#s = sym.get_internals()['fullyconnected0_output']
model1 = mx.module.Module(symbol=sym, context=mx.gpu(0))
model1.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model1.set_params(arg_params=arg_params, aux_params=aux_params)


####################################第二个任务
N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed_4/data/files_2/pysift_train/'

# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed_4/data/model/part2/cnn_sift/100_train_sifts.pkl', 'rb') as point:
    app_train_sifts2 = pickle.load(point) 
# app_train_sifts2 = {}
app_train_names2 = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed_4/data/files_2/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand2 = {}
    for item in brand_temp:
        brand2[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names2:
    app_brand.append(brand2[int(k)])
app_brand_dic2 = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic2:
        app_brand_dic2[app_brand[i]] = [i]
    else:
        app_brand_dic2[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed_4/data/model/part2/myresnext_wd0.00008',26)
#s = sym.get_internals()['fullyconnected0_output']
model2 = mx.module.Module(symbol=sym, context=mx.gpu(1))
model2.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model2.set_params(arg_params=arg_params, aux_params=aux_params)

####################################第三个任务
N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed_4/data/files_3/pysift_train/'

# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed_4/data/model/part3/cnn_sift/100_train_sifts.pkl', 'rb') as point:
    app_train_sifts3 = pickle.load(point) 
# app_train_sifts2 = {}
app_train_names3 = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed_4/data/files_3/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand3 = {}
    for item in brand_temp:
        brand3[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names3:
    app_brand.append(brand3[int(k)])
app_brand_dic3 = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic3:
        app_brand_dic3[app_brand[i]] = [i]
    else:
        app_brand_dic3[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed_4/data/model/part3/myresnext_wd0.00008',28)
#s = sym.get_internals()['fullyconnected0_output']
model3 = mx.module.Module(symbol=sym, context=mx.gpu(1))
model3.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model3.set_params(arg_params=arg_params, aux_params=aux_params)
####################################四个任务
N = 500000
SIFT_ROOT = '/home/wangya/wine/distributed_4/data/files_4/pysift_train/'

# app_train_sifts = load_sift()
with open('/home/wangya/wine/distributed_4/data/model/part4/cnn_sift/100_train_sifts.pkl', 'rb') as point:
    app_train_sifts4 = pickle.load(point) 
# app_train_sifts2 = {}
app_train_names4 = np.array(map(lambda x:x.strip('.pkl'), os.listdir(SIFT_ROOT)))
with open('/home/wangya/wine/distributed_4/data/files_4/train_dic.pkl') as f:
    brand_temp = pickle.load(f)
    brand4 = {}
    for item in brand_temp:
        brand4[np.int64(item)] = brand_temp[item][0]
app_brand = []
for k in app_train_names4:
    app_brand.append(brand4[int(k)])
app_brand_dic4 = {}
for i in range(len(app_brand)):
    if app_brand[i] not in app_brand_dic4:
        app_brand_dic4[app_brand[i]] = [i]
    else:
        app_brand_dic4[app_brand[i]].append(i)

sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/distributed_4/data/model/part4/myresnext_wd0.00008',29)
#s = sym.get_internals()['fullyconnected0_output']
model4 = mx.module.Module(symbol=sym, context=mx.gpu(1))
model4.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
model4.set_params(arg_params=arg_params, aux_params=aux_params)



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

def cnn(job_num, test_img):
    # CNN分类
    global app_train_sifts1, app_train_names1, brand1, app_brand_dic1, model1
    global app_train_sifts2, app_train_names2, brand2, app_brand_dic2, model2
    if job_num == 0:
        model = model1
    elif job_num == 1:
        model = model2
    else:
        raise Exception('Currently, this process is designed for two threads!')
   

    test_image_name = os.path.split(test_img)[-1].split('.')[0]
    # test_img = '/home/wangya/wine/distributed/data/files_' + str(job_num + 1) + '/test/' + test_image_name+'.jpg'

    croped_dir = test_img.replace('/test/' + test_image_name+'.jpg', '/test_crop/' + test_image_name+'.jpg')
    brand=cnn_label.main(job_num, croped_dir, model)
    return brand


def SIFT(test_img,RANGE, THRES0, THRES1, T2):
    global app_train_sifts1, app_train_names1, brand1, app_brand_dic1, model1
    global app_train_sifts2, app_train_names2, brand2, app_brand_dic2, model2
    global app_train_sifts3, app_train_names3, brand3, app_brand_dic3, model3
    global app_train_sifts4, app_train_names4, brand4, app_brand_dic4, model4
    brand11=cnn(0, test_img)
    brand22=cnn(1, test_img)
    brand33=cnn(2, test_img)
    brand44=cnn(3, test_img)
    
    im = cv2.imread(test_img)

    test_image_name = os.path.split(test_img)[-1].split('.')[0]
    mask_file = test_img.replace('/test/' + test_image_name+'.jpg', '/test_mask/' + test_image_name+'.jpg')
    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE) #opencv imread gray image
    
    factor = 500. / max(im.shape)
    im = cv2.resize(im, (int(im.shape[1]*factor), int(im.shape[0]*factor)))

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
    candi1 = app_train_sifts1[reduce(lambda x, y: x+y, [app_brand_dic1[bb] for bb in brand11])]
    candi2 = app_train_sifts2[reduce(lambda x, y: x+y, [app_brand_dic2[bb] for bb in brand22])]
    candi3 = app_train_sifts3[reduce(lambda x, y: x+y, [app_brand_dic3[bb] for bb in brand33])]
    candi4 = app_train_sifts4[reduce(lambda x, y: x+y, [app_brand_dic4[bb] for bb in brand44])]
    candi_1 = np.vstack((candi1, candi2))
    candi_3 = np.vstack((candi3, candi4))
    candi = np.vstack((candi_1, candi_3))
    n_candi = len(candi)
    other_score = Parallel(n_jobs=16)(delayed(get_score2)(ss, des, pos, THRES0, THRES1, T2) for ss in candi)
    other_score = np.array(other_score)


    des_weight = other_score.sum(0)
    des_weight[des_weight==0] = 1      
    des_weight = n_candi / des_weight
    npscore = np.inner(other_score, des_weight)
    best = np.argsort(npscore)[-1:-6:-1]

    
    result_imgs = []
    # app_train_names_sub = app_train_names[[ii for ii in range(len(app_brand)) if app_brand[ii] in brand]]
    app_train_names_sub1 = app_train_names1[reduce(lambda x, y: x+y, [app_brand_dic1[bb] for bb in brand11])]
    app_train_names_sub2 = app_train_names2[reduce(lambda x, y: x+y, [app_brand_dic2[bb] for bb in brand22])]
    app_train_names_sub3 = app_train_names3[reduce(lambda x, y: x+y, [app_brand_dic3[bb] for bb in brand33])]
    app_train_names_sub4 = app_train_names4[reduce(lambda x, y: x+y, [app_brand_dic4[bb] for bb in brand44])]
    app_train_names_sub11 = np.hstack((app_train_names_sub1, app_train_names_sub2))
    app_train_names_sub33 = np.hstack((app_train_names_sub3, app_train_names_sub4))
    app_train_names_sub = np.hstack((app_train_names_sub11, app_train_names_sub33))
    for i in best:
        result_img = app_train_names_sub[i]
        result_imgs.append(result_img)
    # print(app_brand[app_brand == brand][best])
    return result_imgs

if __name__ == '__main__':
   # test_img = './1.jpg'
    RANGE =2000
    THRES0 = 1.5
    THRES1 =5.0
    T2 = 100
    t = 0 #表示时间
    start_time = time.time()
    dir = '/home/wangya/wine/distributed_4/data/files_1/test'
    for img in os.listdir(dir):
        print(img)
        test_img = os.path.join(dir,img)
        result_imgs= SIFT(test_img,RANGE,THRES0,THRES1,T2)
      
        for result_img in result_imgs:
            result_img = result_img + '.jpg'
            result_img1 = os.path.join('/home/wangya/wine/distributed_4/data/files_1/train',result_img)
            result_img2 = os.path.join('/home/wangya/wine/distributed_4/data/files_2/train',result_img)
            if os.path.exists(result_img1):
                read_result.cp_img(test_img,result_img1,'/home/wangya/wine/distributed_4/parallel/result_5brand_5_fa2')
            else:
                read_result.cp_img(test_img,result_img2,'/home/wangya/wine/distributed_4/parallel/result_5brand_5_fa2')
    ti = time.time() - start_time
   
    dir = '/home/wangya/wine/distributed_4/data/files_2/test'
    for img in os.listdir(dir):
        print(img)
        test_img = os.path.join(dir,img)
        result_imgs= SIFT(test_img,RANGE,THRES0,THRES1,T2)
      
        for result_img in result_imgs:
            result_img = result_img + '.jpg'
            result_img1 = os.path.join('/home/wangya/wine/distributed_4/data/files_1/train',result_img)
            result_img2 = os.path.join('/home/wangya/wine/distributed_4/data/files_2/train',result_img)
            if os.path.exists(result_img1):
                read_result.cp_img(test_img,result_img1,'/home/wangya/wine/distributed_4/parallel/result_5brand_5_fa2')
            else:
                read_result.cp_img(test_img,result_img2,'/home/wangya/wine/distributed_4/parallel/result_5brand_5_fa2')
    t = time.time() - ti - start_time
    print('total time is :',ti)   
   
    
    print('total time is :',t)   
   
