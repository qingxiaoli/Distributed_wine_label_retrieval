#!/usr/bin/env python3
import pandas
import random
import os
import pickle


with open('/home/wangya/wine/distributed_4/data/files_4/train.xlsx') as f:
    d = pandas.read_excel(f)

brands = set(d.manufacturer)
print(len(brands)) #the class num of train sunset
idx = 0
bd = {}
for b in brands:
    bd[b] = idx 
     # the index of label ,the dic bd save the relation between real name of label and index of label 
    idx += 1    
#put label and label index write into file
with open('/home/wangya/wine/distributed_4/data/files_4/train_label_index.pkl', 'wb') as output:
    pickle.dump(bd, output) 

with open('/home/wangya/wine/distributed_4/data/files_4/train_dic.pkl', 'rb') as point:
    image_inf = pickle.load(point)

to_write = []
idx = 0
for image in os.listdir('/home/wangya/wine/distributed_4/data/files_4/aug_train_crop/'):
    im_id = image.split('_')[0]
    to_write.append('%d\t%d\t/home/wangya/wine/distributed_4/data/files_4/aug_train_crop/%s\n'%(idx, bd[image_inf[im_id][0]],image))
    idx += 1



print(idx) #this is the all num of data in the train subset
random.shuffle(to_write) # make random for traindata

with open('/home/wangya/wine/distributed_4/data/files_4/train.lst', 'w') as f:
    for line in to_write:
        f.write(line)
print('Congratulation! The progess finished!')