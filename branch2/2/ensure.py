import os

dir1 = '/home/wangya/wine/distributed/2/result_5brand_5'
dir2 = '/home/wangya/wine/distributed/2/result_5brand_51'
cha = list(set(os.listdir(dir1))-set(os.listdir(dir2)))
print(len(cha))

print(len(os.listdir(dir2)))
print(len(os.listdir(dir1)))