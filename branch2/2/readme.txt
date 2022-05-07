对第二部分的整体测试框架用了两种方案：
方案1：
cnn得到5个主品牌，这5个主品牌的所有样本混合在一起进行sift 匹配，最后返回前5个
程序为： sift_cnn_label_fast_5brand.py
得到的结果在 ./result_5brand_5/下

时间为：15s/pic
精度为： brand_acc: 92.58%  item_acc=81.05%


















方案2：
cnn得到5个主品牌，每个主品牌的样本进行sift匹配，并返回一个最大匹配的样本，所以共5个返回
程序为：sift_cnn_label_fast_5brand5.py
得到的结果保存在：./result_5brand_51/下
时间为：19s/pic
精度为： brand_acc: 93.45%  item_acc=63.16%



