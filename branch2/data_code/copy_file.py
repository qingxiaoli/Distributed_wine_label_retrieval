#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import shutil

train_files = '../data/train/'
aug_files = '../data/aug_data/'

files = os.listdir(train_files)
num = len(files)
i = 0
for file in files:
	full_file = os.path.join(train_files, file)
	file_id = file.split('.')[0]
	new_file = os.path.join(aug_files, file_id + '_-1_raw_0.jpg')
	if not os.path.exists(new_file):
		shutil.copy(full_file, new_file)
		i += 1
		print('file:{} \t total:{}'.format(i, num))
