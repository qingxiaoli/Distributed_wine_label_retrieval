# -*- coding:utf-8 -*-

"""
这个程序主要生成mask图片，用于seg,愚蠢的我啊，replace那个地方还是错的！！！！！
"""
# pylint: skip-file



import cPickle as pickle
import numpy as np
import mxnet as mx
import os
import cv2
import urllib
import cStringIO

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True #因为部分原图会出现'truncated'现象，即有元素值超过阈值，
                                       #加载这类图片需要添加左边的程序，特此声明！！！

pallete = [ 0,0,0,
            255,255,255 ]
model_previx = "./model_pascal/FCN8s_VGG16"
epoch = 2
ctx = mx.gpu(0)

def get_data(img):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = np.array(img, dtype=np.float32)
    if img.shape[0] > img.shape[1]:
        old_shape = (int(img.shape[1]*500./img.shape[0]), 500)
    else:
        old_shape = (500, int(img.shape[1]*500./img.shape[0]))
    img = cv2.resize(img, (500,500))
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img, old_shape


def do(x,n):
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    files_dir = '/home/wangya/wine/data_augment/data/test'
    num = x
    for file in os.listdir(files_dir)[x:n]:
        print(num)
        print(file)
        file_dir = os.path.join(files_dir,file)
        #save_name = file.split('.')[0]+'.png'
        save_name = file.replace('.png', '.jpg')
        num = num+1
        if os.path.exists('/home/wangya/wine/data_augment/data/test_mask/%s'%save_name):
            continue
        im = Image.open(file_dir).convert('RGB') 
        img, old_shape = get_data(im) 
        fcnxs_args["data"] = mx.nd.array(img, ctx)
        data_shape = fcnxs_args["data"].shape
        label_shape = (1, data_shape[2]*data_shape[3])
        fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
        exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_args)
        exector.forward(is_train=False)
        output = exector.outputs[0]
        out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
        out_img = cv2.resize(out_img, old_shape, interpolation=cv2.INTER_NEAREST)
        out_img = Image.fromarray(out_img)
        out_img.putpalette(pallete)
        out_img = out_img.convert('RGB')
        out_img.save('/home/wangya/wine/data_augment/data/test_mask/%s'%save_name)
     
if __name__ == "__main__":
    do(30000,70000)
