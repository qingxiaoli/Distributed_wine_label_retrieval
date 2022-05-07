import cv2
import numpy as np
import os

def do(x, n):
    # pics = sorted(os.listdir('test_mask'))
    pics = os.listdir('/home/wangya/wine/distributed_4/data/files_2/train_mask')
    num = x
    none = 0
    for name in pics[x:x+n]:
        print(num)
        num = num+1
        print(name) 
        if os.path.exists('/home/wangya/wine/distributed_4/data/files_2/train_crop/'+name):
            continue
        im = cv2.imread('/home/wangya/wine/distributed_4/data/files_2/train/'+name)
        # while True:
        mask = cv2.imread('/home/wangya/wine/distributed_4/data/files_2/train_mask/'+name)
        if mask is None:
            none =none+1
            
            continue
        if im.shape != mask.shape:
            print im.shape
            print mask.shape
        x0, y0 = 0, 0
        x1, y1, _ = mask.shape
        x1 -= 1
        y1 -= 1
        try:
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
            cv2.imwrite('/home/wangya/wine/distributed_4/data/files_2/train_crop/'+name, im)
        except:
            print name
            raise Exception
    print(none)
do(0,1000000)