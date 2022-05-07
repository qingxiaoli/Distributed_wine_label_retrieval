save_pysift.py 是对虽有图像提取sift描述子，其中主要的函数为cv2.siftdectectandcompute(img,mask)
意思是对img中的mask区域部分提取sift描述子。
对data/file_1里面的train和test 里面的图像都要提取，分别放在pysift_train和pysift_test文件夹下
sift_pkl.py是生成对所有的sift进行 load的一个暂存文件，节约整个的读取时间。

sift_cnn_label_fast_5brand.py：
 1、 是指对测试数据整个流程跑一遍，得到检索结果（每张测试图像返回5个主品牌，
对这5个主品牌所有图像一起进行sift匹配返回5张）
 2、score5.py
对sift_cnn_label_fast_5brand.py生成的检索结果进行计算准确率。
 3、result_5brand_5 这是检索结果
 avg_time:2.5932s     brand_acc: 90.60%      item_acc:85.16%



sift_cnn_label_fast_5brand5.py：
对返回的5个主品牌，每个主品牌进行一次sift匹配，且返回一个最匹配的样本，一共进行5次匹配
结果存放在：./result_5brand_51/下：

avg_time:2.7152s    brand_acc:91.15%   iteam_acc: 69.83%