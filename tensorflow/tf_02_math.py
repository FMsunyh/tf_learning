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

def test_rpn():
    a = tf.range(16)
    a = tf.reshape(a, shape=(2,2,4))

    b = tf.reshape(a, shape=(-1,))
    c = tf.reshape(a, shape=(-1,1))

    a = tf.expand_dims(a, axis=0)
    sum = tf.reduce_sum(a, axis=[1,2,3])
    ave = tf.reduce_mean(sum)
    # tf.Print(sum, [sum], 'sum', summarize=20)
    tf.Print(ave, [ave], 'ave', summarize=20)

def test_rcnn():
    a = tf.ones(shape=(16,))
    a = tf.reshape(a, shape=(2,2*4))

    a = tf.reshape(a, shape=(-1,1))
    a = tf.expand_dims(a, axis=0)
    sum = tf.reduce_sum(a, axis=[1])
    ave = tf.reduce_mean(sum)
    # tf.Print(sum, [sum], 'sum', summarize=20)
    tf.Print(ave, [ave], 'ave', summarize=20)


def proposal():
    a = tf.range(10, 100)
    a = tf.reshape(a, shape=(-1, 1))
    _,indices = tf.nn.top_k(a[:,0], k=10)

    c = tf.gather(a, indices)
    print(indices)


def test_where():
    a = np.array([-1,1,0,1,1])
    labels = tf.constant(a, dtype=tf.int32)
    c = tf.constant([1], dtype=tf.int32)
    indices = tf.where(tf.greater_equal(labels, 1))[:,0]
    debug = tf.gather(labels, indices)
    labels = tf.Print(labels, [labels[1:20], tf.gather(labels, indices)[1:20]], 'labels', summarize=10)

def _inside_image(boxes):
    """
    Filter anchor which coordinates overflow input image shape
    :param boxes: (N, 4) ndarray of float
    :param image_shape: [H,W]
    :return: indices: anchors index which inside image,shape = (A,1) A<=N
    :return: indices: anchors coordinates which inside image,shape = (A,4) A<=N
    """
    allowed_border = 0
    w = 600.
    h = 400.

    indices = tf.where(
        (boxes[:, 0] >= -allowed_border) &
        (boxes[:, 1] >= -allowed_border) &
        (boxes[:, 2] < allowed_border + w) &  # width
        (boxes[:, 3] < allowed_border + h)  # height
    )

    indices = tf.to_int32(indices)[:, 0]
    inside_boxes = tf.gather(boxes, indices)
    return indices[:, 0], tf.reshape(inside_boxes, [-1, 4])


def _roi_pooling():
    """
    Filter anchor which coordinates overflow input image shape
    :param boxes: (N, 4) ndarray of float
    :param image_shape: [H,W]
    :return: indices: anchors index which inside image,shape = (A,1) A<=N
    :return: indices: anchors coordinates which inside image,shape = (A,4) A<=N
    """
    boxes = np.array([[10, 10, 20, 20], [-10, 10, -20, 30]])
    rois = tf.cast(boxes, tf.float32)
    x1 = rois[..., 0]
    y1 = rois[..., 1]
    x2 = rois[..., 2]
    y2 = rois[..., 3]

    rois = rois / 10

    x1 = tf.expand_dims(x1, axis=-1)
    y1 = tf.expand_dims(y1, axis=-1)
    x2 = tf.expand_dims(x2, axis=-1)
    y2 = tf.expand_dims(y2, axis=-1)

    # rois = tf.concatenate([x1, y1, x2, y2], axis=-1)
    rois = tf.concat([y1, x1, y2, x2], axis=-1)
    rois = tf.reshape(rois, (-1, 4))

    rois /=10
    print(rois)


def sort():
    a = tf.constant([0.144317135,0.202581272,0.509946227,0.424173057,0.243863285,0.168587461,0.426486045,0.587834656,
                     0.223706454,0.337008804,0.374025494,0.263728261,0.187421843,0.379993,0.188351691,0.597955406,
                     0.305669874,0.367155939,0.193560496,0.369595468,0.400290847,0.202153459,0.264195263,
                     0.316051304,0.396186084,0.274192333,0.48402375,0.530695856,0.478960305,0.515173078,0.387901634,
                     0.41583702,0.500061691,0.478882939,0.40127933,0.381851286,0.328007966,0.449395329])

    print(a)

    indices = tf.where(a > 0.4)[:,0]
    a = tf.gather(a,indices)

    print(a)

    v,i =  tf.nn.top_k(a,k=tf.size(a))
    sort_indices= tf.gather(indices,i)

    print(v)
    print(i)
    print(sort_indices)


def argmax():
    boxes = np.array([[10, 10, 20, 20], [-10, 10, -20, 30]])
    rois = tf.cast(boxes, tf.float32)
    a = tf.argmax(rois, axis=0)
    print(a)

if __name__ == '__main__':
    # test_rpn()
    # test_rcnn()
    # proposal()
    # test_where()
    # boxes = np.array([[10,10, 20,20],[-10,10, 20,20]])
    # a,b = _inside_image(boxes)
    # print(a,b)

    # a = tf.constant([[[0.1,0.9]]])
    # a = tf.tile(a, [1,1,9])
    # c = tf.reshape(a, shape=(1, 9, 2))
    # print(a)
    # print(c[0, :, :])

    # print(c[0, 0, 9:])
    # print(a)
    # _roi_pooling()

    # sort()
    argmax()