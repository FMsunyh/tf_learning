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
# @Time    : 12/11/2018 9:36 AM
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

def tile_test():
    value = tf.constant([[1,2,3],[4,5,6]], dtype=tf.int32)

    value_tile = tf.tile(value, [1,2])
    tf.Print(value, [tf.shape(value)], 'value:', summarize=20)
    tf.Print(value_tile, [value_tile], 'value_tile:', summarize=20)

    value_tile = tf.tile(value, [2,2])
    tf.Print(value, [tf.shape(value)], 'value:', summarize=200)
    tf.Print(value_tile, [value_tile], 'value_tile:', summarize=200)


def test():
    cell_size = 3
    boxes_per_cell =2
    # offset = np.transpose(np.reshape(np.array([np.arange(cell_size)] * cell_size * boxes_per_cell),(boxes_per_cell, cell_size, cell_size)), (1, 2, 0))
    offset =np.reshape(np.array([np.arange(cell_size)] * cell_size * boxes_per_cell),(boxes_per_cell, cell_size, cell_size))
    tf.Print(offset, [offset], 'offset:', summarize=200)
    tf.Print(offset, [tf.shape(offset)], 'offset shape:', summarize=200)

    offset = tf.reshape(offset, [1, cell_size, cell_size, boxes_per_cell])
    tf.Print(offset, [offset], 'offset:', summarize=200)
    tf.Print(offset, [tf.shape(offset)], 'offset shape:', summarize=200)

    offset = tf.tile(offset, [1, 1, 1, 1])

    tf.Print(offset, [offset], 'offset:', summarize=200)
    tf.Print(offset, [tf.shape(offset)], 'offset shape:', summarize=200)


def scatter_add_tensor(ref, indices, updates, name=None):
    """
    Adds sparse updates to a variable reference.

    This operation outputs ref after the update is done. This makes it
    easier to chain operations that need to use the reset value.

    Duplicate indices: if multiple indices reference the same location,
    their contributions add.

    Requires updates.shape = indices.shape + ref.shape[1:].

    :param ref: A Tensor. Must be one of the following types: float32,
    float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128,
    qint8, quint8, qint32, half.

    :param indices: A Tensor. Must be one of the following types: int32,
    int64. A tensor of indices into the first dimension of ref.

    :param updates: A Tensor. Must have the same dtype as ref. A tensor of
    updated values to add to ref

    :param name: A name for the operation (optional).

    :return: Same as ref. Returned as a convenience for operations that want
    to use the updated values after the update is done.
    """
    with tf.name_scope(name, 'scatter_add_tensor', [ref, indices, updates]) as scope:
        # ref = tf.convert_to_tensor(ref, name='ref')
        # indices = tf.convert_to_tensor(indices, name='indices')
        # updates = tf.convert_to_tensor(updates, name='updates')
        ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
        scattered_updates = tf.scatter_nd(indices, updates, ref_shape, name='scattered_updates')
        with tf.control_dependencies(
                [tf.assert_equal(ref_shape, tf.shape(scattered_updates, out_type=indices.dtype))]):
            output = tf.add(ref, scattered_updates, name=scope)

        return output

def test_scatter():
    labels = tf.ones(shape=(100,))
    labels = tf.concat([labels, tf.ones(shape=(28,))* 2], axis=0)

    update_indices = tf.range(100, tf.shape(labels)[0])
    update_indices = tf.reshape(update_indices, (-1, 1))
    inverse_labels = tf.gather_nd(labels, update_indices) * -2
    labels = scatter_add_tensor(labels, update_indices, inverse_labels)

    labels = tf.Print(labels, [labels], 'labels', summarize=20)

def _umap_bbox_tranfrom(labels, bbox_target):
    num_classes = 21
    positive_index = tf.reshape(tf.where(tf.greater_equal(labels, 1)), shape=[-1])
    positive_cls = tf.to_int64(tf.gather(labels, positive_index))

    positive_bbox_target = tf.gather(bbox_target, positive_index)
    indices = tf.reshape(tf.to_int32(positive_index * num_classes + positive_cls), shape=[-1, 1])
    bbox_regression_targets = tf.scatter_nd(indices, positive_bbox_target, shape=[tf.size(labels) * num_classes, 4])
    bbox_regression_targets = tf.to_float(tf.reshape(bbox_regression_targets, shape=[-1, num_classes * 4]))

    return bbox_regression_targets

def test_umap_bbox_tranfrom():
    labels = tf.range(1,3)
    labels = tf.concat([labels, tf.zeros(shape=(10,), dtype=tf.int32)], axis=0)

    box = np.zeros((1, 4))
    box[0, 0] = 10
    box[0, 1] = 10
    box[0, 2] = 10
    box[0, 3] = 10
    boxes = []
    boxes.append(box)
    boxes.append(box*4)

    for _ in range(10):
        boxes.append(box * 0)

    bbox_target = tf.constant(boxes)

    bbox_target = tf.reshape(bbox_target, shape=(-1, 4))
    s = _umap_bbox_tranfrom(labels, bbox_target)

    s = tf.reshape(s, shape=(12, 21, 4))

    print('')
if __name__ == '__main__':
    test_umap_bbox_tranfrom()