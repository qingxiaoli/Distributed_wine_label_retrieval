# -*- coding: utf-8 -*-
import mxnet as mx
import logging
#import pdb

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


s,arg_params,aux_params = mx.model.load_checkpoint('./myresnext_wd0.00015',8)
batch_size =100

val_rec = mx.io.ImageRecordIter(
        path_imgrec = '/home/wangya/wine/data_augment/data/test.rec',
        #mean_img    = 'mean.nd',
        data_shape  = (3, 192, 128),
        preprocess_threads = 16,
        batch_size  = batch_size,
        rand_crop   = False,
        rand_mirror = False
    )



model = mx.module.Module(symbol=s, context=mx.gpu())
model.bind(for_training=False,
             data_shapes=val_rec.provide_data,
             label_shapes=val_rec.provide_label)
model.set_params(arg_params=arg_params, aux_params=aux_params)

# metrics = [mx.metric.create('acc'),
#                mx.metric.create('top_k_accuracy', top_k = 5)]
# num = 0
# for batch in val_rec:
#     model.forward(batch, is_train=False)
#     for m in metrics:
#         model.update_metric(m, batch.label)
#     num += batch_size     #####就是很不清楚这里的是针对所有的test来的吗？每个函数的原理是什么
# for m in metrics:
#     logging.info(m.get())
score = model.score(val_rec,['mse','acc'])
print(score)

# #下面是对于内部权重的输出
keys = model.get_params()[0].keys() # 列出所有权重名称
print(keys)
# #conv_w = model.get_params()[0]['fullyconnected0_weight'] #获取想要查看的权重信息,如conv_weight
# #temp = conv_w.asnumpy()
# #pdb.set_trace()
# #print conv_w.asnumpy()[0]

# #下面是对与中间结果的输出
# batch_size = 1
# val_rec = mx.io.ImageRecordIter(
#         path_imgrec = 'test.rec',
#         #mean_img    = 'mean.nd',
#         data_shape  = (3, 192, 128),
#         preprocess_threads = 16,
#         batch_size  = batch_size,
#         rand_crop   = False,
#         rand_mirror = False
#     )

# args = s.get_internals().list_outputs() #获得所有中间输出
# internals = model.symbol.get_internals()
# sft = internals['fullyconnected0_output']
# mod = mx.module.Module(symbol=sft, context=mx.gpu()) #创建Module
# mod.bind(for_training=False, data_shapes=val_rec.provide_data) #绑定，此代码为预测代码，所以training参数设为False
# mod.set_params(arg_params, aux_params)

# for batch in val_rec:
#     mod.forward(batch) #预测结果
#     prob = mod.get_outputs()[0].asnumpy()
#     print(prob)
#     break
