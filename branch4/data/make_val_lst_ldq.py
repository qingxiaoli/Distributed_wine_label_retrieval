#this can transform the ori data and labl to lst format,and this is for validation data and label
#ouput: 1,the .lst file for generate the rec format file   ,no have 2
#the diff with make_train_lst_lq.py is that the index of label must be same as the train, indepedencd with the pair train 
import pandas
import random
import os
import pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# brands = set(d.manufacturer)
# print(len(brands)) #the class num of train sunset
# idx = 0
# bd = {}
# for b in brands:
#     bd[b] = idx 
#      # the index of label ,the dic bd save the relation between real name of label and index of label 
#     idx += 1    
# #put label and label index write into file
# output = open('train1_label_index.pkl', 'wb')
# pickle.dump(bd, output)
# output.close() 


#load the pre saved file
with open('/home/wangya/wine/distributed_4/data/files_4/train_label_index.pkl', 'rb') as pkl_file:
	data1 = pickle.load(pkl_file)

with open('/home/wangya/wine/distributed_4/data/files_4/test_dic.pkl', 'rb') as point:
    image_inf = pickle.load(point)

to_write = []
idx = 0
for image in os.listdir('/home/wangya/wine/distributed_4/data/files_4/test_crop/'):
    im_id = image.split('.')[0]
    to_write.append('%d\t%d\t/home/wangya/wine/distributed_4/data/files_4/test_crop/%s\n'%(idx, data1[image_inf[im_id][0]],image))
    idx += 1

print(idx) #this is the all num of data in the train subset
random.shuffle(to_write) # make random for traindata

with open('/home/wangya/wine/distributed_4/data/files_4/test.lst', 'w') as f:
    for line in to_write:
        f.write(line)


        
