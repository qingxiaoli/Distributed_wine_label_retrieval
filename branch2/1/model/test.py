import sys
sys.path.append('/home/wangya/wine/model/incubator-mxnet/example/image-classification/')
import argparse
from common import modelzoo, find_mxnet
import mxnet as mx
import time
import os
import logging

c2 = mx.sym.loads('./myresnext_wd0.00015-symbol.json')
print(c2)