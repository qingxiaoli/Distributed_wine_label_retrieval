import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time


def list_image(root, recursive, exts):
    image_list = []
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root, followlinks=True):
            subdirs.sort()
            print(len(cat), path)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    image_list.append((len(image_list), os.path.relpath(fpath, root), cat[path]))
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                image_list.append((len(image_list), os.path.relpath(fpath, root), 0))
    return image_list


def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        n_images = xrange(len(image_list))
        for i in n_images:
            line = '%d\t' % image_list[i][0]
            for j in image_list[i][2:]:
                line += '%d\t' % j
            line += '%s\n' % image_list[i][1]
            fout.write(line)



def read_list(path_in):
    image_list = []
    with open(path_in) as fin:
        for line in fin.readlines():
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [int(i) for i in line[1:-1]]
            image_list.append(item)
    return image_list


def write_record(image_list, fname):
    source = image_list
    tic = [time.time()]
    color_modes = {-1: cv2.IMREAD_UNCHANGED,
                   0: cv2.IMREAD_GRAYSCALE,
                   1: cv2.IMREAD_COLOR}
    total = len(source)

    def image_encode(item, q_out):
        try:
            img = cv2.imread(os.path.join('.', item[1]), color_modes[1])
        except:
            print 'imread error:', item[1]
            return
        if img is None:
            print 'read none error:', item[1]
            return
        img = cv2.resize(img, (128,192))
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

        try:
            s = mx.recordio.pack_img(header, img, quality=100, img_fmt='.jpg')
            q_out.put(('data', s, item))
        except:
            print 'pack_img error:', item[1]
            return

    def read_worker(q_in, q_out):
        while not q_in.empty():
            item = q_in.get()
            image_encode(item, q_out)

    def write_worker(q_out, fname, saving_folder):
        pre_time = time.time()
        sink = []
        os.chdir(saving_folder)
        fname_rec = fname[:fname.rfind('.')]
        record = mx.recordio.MXRecordIO(fname_rec + '.rec', 'w')
        while True:
            stat, s, item = q_out.get()
            if stat == 'finish':
                write_list(fname_rec + '.lst', sink)
                break
            record.write(s)
            sink.append(item)
            if len(sink) % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', len(sink)
                pre_time = cur_time

    try:
        import multiprocessing
        q_in = [multiprocessing.Queue() for i in range(16)]
        q_out = multiprocessing.Queue(1024)
        for i in range(len(image_list)):
            q_in[i % len(q_in)].put(image_list[i])
        read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) \
                        for i in range(16)]
        for p in read_process:
            p.start()
        write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, '.'))
        write_process.start()
        for p in read_process:
            p.join()
        q_out.put(('finish', '', []))
        write_process.join()
    except ImportError:
        print('multiprocessing not available, fall back to single threaded encoding')
        import Queue
        q_out = Queue.Queue()
        os.chdir('.')
        fname_rec = fname[:fname.rfind('.')]
        record = mx.recordio.MXRecordIO(fname_rec + '.rec', 'w')
        cnt = 0
        pre_time = time.time()
        for item in image_list:
            image_encode(item, q_out)
            if q_out.empty():
                continue
            _, s, _ = q_out.get()
            record.write(s)
            cnt += 1
            if cnt % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', cnt
                pre_time = cur_time


def main(f):
    image_list = read_list(f)
    write_record(image_list, f)


if __name__ == '__main__':
    main('/home/wangya/wine/distributed_4/data/files_4/train.lst')
    main('/home/wangya/wine/distributed_4/data/files_4/test.lst')
    #main('/home/wangya/wine/distributed/data/files_2/test.lst')
  
