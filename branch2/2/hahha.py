import os


dir1 = '/home/wangya/wine/distributed/data/files_2/test'
dir2 = '/home/wangya/wine/distributed/2/result_5brand_5'
aa = '20286501.jpg'
a = os.listdir(dir1)
# b = os.listdir(dir2)
# if a in b:
#     print('sure') 
# else:
# 	print('heng')
dd = a.index(aa)
print(dd)