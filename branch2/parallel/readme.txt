parallel_file1.py:
	两个子网络，每个子网络之后用一个“合并”SIFT匹配返回5张待选图像。
	子网络放在GPU，不并行，每个SIFT匹配用16个线程并行。

	整体架构：
		子网络1 -> 并行SIFT匹配 -> 子网络2 -> 并行SIFT匹配

	测试集：
		/home/wangya/wine/distributed/data/files_1/test

	GPU使用：
		1,2号

	实验结果：
		总时间：208545.50s
		平均时间：9.85s


parallel_file2.py:
	两个子网络，每个子网络之后用一个“合并”SIFT匹配返回5张待选图像。
	子网络放在GPU，不并行，每个SIFT匹配用16个线程并行。

	整体架构：
		子网络1 -> 并行SIFT匹配 -> 子网络2 -> 并行SIFT匹配

	测试集：
		/home/wangya/wine/distributed/data/files_2/test

	GPU使用：
		1,2号
result_5brand_5/:
	parallel_file1.py和parallel_file2.py的返回结果。

brand_acc   item  -acc

0.9217      0.8194








cnn_label_yd5.py:
	在非并行版的基础上改了一下，根据当前的线程号选择用哪个train_label_index.pkl文件。



parallel2.py:  
网络1---网络2---sift匹配
	测试集：
		/home/wangya/wine/distributed/data/files_1/test
		/home/wangya/wine/distributed/data/files_2/test

	GPU使用：
		2,3号

结果保存在result_5brand_5_fa2里面

brand_acc   item  -acc
0.9181       0.8108

