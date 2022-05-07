/home/wangya/wine/distributed_4/data
下的div_data_1.py和div_data_2.py是对数据进行分类
直接是对/home/wangya/wine/distributed_2/data的进行分类，也就是他这里面的file_1数据分成这里的两部分，file_2也是。

div_data_tmp.py是生产保留中间结果的pkl文件，为了数据分部分比较方便

本文件夹下
                                样本量           main_brand           sub_brand      总图像量      

files_1 数据类样本： 11~14           4943                        33941          60908
files_2 数据类样本： 14~20           4031                        36724          69433
files_3 数据类样本： 21~34           4218                        55615          111425
files_4数据类样本：  34~1371         4136                       140415         306091


上面都是 将数据分为训练集和测试集，两部分比例为8:2，用到的代码为： ./train_test_segfirst.py
               test          train样本量    train主品牌数目   train子品牌数目         aug_train样本量
files_1          9193         51715            4943                 33941             (140:140)  692020
files_2          11977        57456            4031                 36724             (200:200)  802169
files_3          19946        91479            4218                 55615             (340:340)  1349760
files_4          58963        247128           4136                 140415           （548：2742）2959646

对每个文件夹下的train里面的图片进行数据增强 用到的程序是 data_augment_1.py.......

接下来对生成的aug_train进行get_mask 和 crop处理
用到的程序是每个文件夹下的get_mask.py和 crop_img.py  存于每个文件夹下的aug_train_mask 和 aug_train_crop 文件夹下
对test 下的图像做同样的处理，得到的存于 test_mask 和 test_crop下


经过CNN_SIFT后每部分的结果如下:
part       brand      item
1          0.9180      0.8694
2          0.9356      0.8691
3          0.9254      0.8524
4          0.9304      0.8021

合起来的精度情况是：
              brand    item
fa1        0.9280    0.8448
fa2        0.9172    0.8228




