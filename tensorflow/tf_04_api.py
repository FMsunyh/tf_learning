#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 11/21/2018 10:00 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

# 1. GPU setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    return tf.Session(config=cfg)

get_session()

tfe.enable_eager_execution()
tfe.executing_eagerly()        # => True

if __name__ == '__main__':
    # labels = tf.constant([1,0,0,0],dtype=tf.int32)
    # indices = tf.constant([2,2], dtype=tf.int32)
    #
    # foreground = tf.ones(tf.shape(labels), dtype=tf.int32)
    #
    # # condition = tf.one_hot(indices, depth=tf.shape(labels))
    # print(tf.shape(labels)[0])
    #
    # condition = tf.one_hot(indices, depth=tf.shape(labels)[0])
    # print(condition)
    #
    # condition = tf.sparse_to_dense(indices, tf.shape(labels, out_type=tf.int32), True, default_value=False, validate_indices=False)
    # print(condition)
    #
    # labels = tf.where(condition=condition, x=foreground, y=labels)
    # print(labels)


    # labels = tf.constant([0,0,0,0],dtype=tf.int32)
    # foreground = tf.ones(tf.shape(labels), dtype=tf.int32)
    # overlaps = tf.constant([[0.3,0.2,0.5,0.7], [0.3,0.6,0.8,0.2], [0.3,0.2,0.5,0.1],[0.3,0.2,0.5,0.1]],dtype=tf.float32)
    # max_overlaps = tf.reduce_max(overlaps, axis=1)
    # labels = tf.where(tf.greater_equal(max_overlaps, 0.7), x=foreground, y=labels)
    # print(labels)

    # # condition = tf.sparse_to_dense(indices, tf.shape(labels, out_type=tf.int32), True, default_value=False)

    # sess = get_session()
    # # condition = tf.Variable([1,0,0,0],dtype=tf.int32)
    # # b = tf.Variable(1,dtype=tf.int32)
    # # # condition = tf.Variable([1,0,0,0],dtype=tf.int32)
    # # a = tf.assign(b,13)
    # # # condition= condition[1].assign(1)
    # # print(a.eval(session=sess))
    #
    # condition = tf.Variable([1,0,0,0],dtype=tf.int32)
    # index = tf.constant([1,2],dtype=tf.int32)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # condition = condition[index[1]].assign(1)
    #
    # print(condition.eval(session=sess))

    # labels = tf.constant([1, 1, 0, 0,-1], dtype=tf.int32)
    # bg_inds = tf.reshape(tf.where(tf.equal(labels, 0)), shape=[-1])
    #
    # indices = tf.where(tf.equal(labels, 0))[:,0]
    # # labels = tf.gather(labels, indices)
    #
    # print(bg_inds)
    # print(indices)
    # print(tf.equal(0.00000001, 0))

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(13), [13]), (1, 13, 13, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [2, 1, 1, 5, 1])

    # # print(cell_grid)
    # a = tf.constant([1,2], dtype=tf.float32)
    # b = tf.constant([2,2], dtype=tf.float32)
    # # print(tf.truediv(a, b))
    #
    # a = tf.constant([[1,0],[0,1],[0,1],[0,1],[0,0]], dtype=tf.float32)
    # # a = tf.expand_dims(a, axis=-1)
    # # print(a)
    #
    # true_box_class = tf.argmax(a[..., :], -1)

    a = tf.constant([[0,0,0,0,0,0.1],[0,0.2,0.12,0,0,0]] , dtype=tf.float32)
    b = tf.constant([[1],[0]], dtype=tf.float32)
    print(tf.shape(a))
    print(tf.shape(b))
    print(tf.where(tf.reduce_sum(a*b,axis=-1) ))
    print(tf.where(b))
    print(a*b)
    
