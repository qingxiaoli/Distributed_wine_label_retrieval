div_data_yd.py:
	将/home/wangya/wine/data_augment/data/cut10_pics/中的数据分为两部分
.
	
分别存放在
		
/home/wangya/wine/distributed/data/files_1/data/
	
和
		
/home/wangya/wine/distributed/data/files_2/data/

	
前者的标签为：
		
/home/wangya/wine/distributed/data/files_1/data.xlsx
	
后者的标签为：
		
/home/wangya/wine/distributed/data/files_2/data.xlsx
	

	
/home/wangya/wine/data_augment/data/cut10_pics/：
		
总数据量：547857
		
类别数量：17328
		
每类所含样本量：11～1371
	

/home/wangya/wine/distributed/data/files_1/data/：
		



总数据量：130341
		
类别数量：8974
		
每类所含样本量：11～20
	

/home/wangya/wine/distributed/data/files_2/data/：
		
总数据量：417516
		
类别数量：8354
		
每类所含样本量：21～1371

	
问题：
		

/home/wangya/wine/distributed/data/files_1/data.xlsx 含130343（>130341）条记录，
缺失两个样本；
		
/home/wangya/wine/distributed/data/files_2/data.xlsx 含417518（>417516）条记录，
缺失两个样本



下面对一部分 file1进行实验
1。将数据分为训练集和测试集  ../code/train_test_segfirst.py
测试与训练的比例是 2：8

数据增强：../code/data_augment_1.py
因为 file_1 下类样本量都在11~20，所以每类样本都增强后的数量为200
训练数据：
测试数据：    













下面对一部分 file2进行实验
1。将数据分为训练集和测试集  ../code/train_test_segfirst.py
测试与训练的比例是 2：8

数据增强：../code/data_augment_1.py
因为 file_2 下类样本量都在20~1600，增强策略类似于没分两部分之前的，样本量在1600的增强2倍，为3200张，20的类增强后为3200/8=400张，，有8倍的数据差

训练数据：
测试数据：



