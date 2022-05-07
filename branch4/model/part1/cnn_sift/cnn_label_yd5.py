# pylint: skip-file
# -*- coding: utf-8 -*-
#不用 mean
import cPickle as pickle
import numpy as np
import mxnet as mx
import os
import cv2
from collections import namedtuple


def get_data(im):
    im = cv2.imread(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    # im=float(im)
    im = map(lambda a1: map(lambda a2: map(lambda a3: float(a3), a2), a1), im)
    #im -= mean
    im = np.expand_dims(im, 0)
    return im



def main(croped_img, model):

    from collections import namedtuple
   # app_mean = mx.nd.load('./mean.nd')['mean_img'].asnumpy()
    im = croped_img
    im = get_data(im)
    
    Batch = namedtuple('Batch', ['data'])
    model.forward(Batch([mx.nd.array(im)]))
    #feat1 = model.get_outputs()
    label_info = model.get_outputs()[0].asnumpy()
    prob = np.squeeze(label_info)
    prob =np.argsort(prob)[::-1]
    label_index_list = [prob[0],prob[1],prob[2],prob[3],prob[4]]

    pkl_file = open('/home/wangya/wine/distributed_4/data/files_1/train_label_index.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    data = range(len(data1.keys()))
    for i in data1:
        data[data1[i]]=i
    label_list=[data[label_index] for label_index in label_index_list]
    print(label_list)
    #return label_index
    #label = label_info.index(max(label_info[0]))
    return label_list
    # feat = feat.flatten()
    # test_img_name = os.path.split(croped_img)[-1].split('.')[0] 
    # feat_dir = os.path.join('./testcnn_feat',test_img_name) +'.pkl'
    # with open(feat_dir, "wb") as f:
    #     pickle.dump(feat,f)
    # return feat  #repr() 函数将对象转化为供解释器读取的形式

if __name__ == "__main__":
    sym, arg_params, aux_params = mx.model.load_checkpoint('/home/wangya/wine/model/resnext50wd01',2)
#s = sym.get_internals()['fullyconnected0_output']
    model = mx.module.Module(symbol=sym, context=mx.gpu())
    model.bind(for_training=False,data_shapes=[('data',(1,3,192,128))])
    model.set_params(arg_params=arg_params, aux_params=aux_params)

    test_dir = '/home/wangya/wine/data_augment/data/test'
    for file in os.listdir(test_dir):
        file_dir = os.path.join(test_dir,file)
        print(file_dir)
        feat = main(file_dir,model)
        break
