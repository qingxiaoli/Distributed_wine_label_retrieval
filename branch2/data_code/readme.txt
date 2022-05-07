copy_file.py:
	把../data/train/里的图像在文件名的末尾添加‘_-1_raw_0’，复制到../data/aug_data/中

cut10.py:
	从../data/80w_pics/中所含样本量大于10的主品牌，对应的图像放在../data/cut10_pics/中

data_augment.py:
	数据增强

test_xlsx:
	检查cut10.xlsx文件的内容是否合理

train_test_segfirst.py:
	从../data/cut10_pics/中划分训练集和测试集

subbrand_num.py:
	统计所有子品牌所含样本量，结果保存在/home/wangya/wine/data_augment/data/subbrand_num.pkl中

