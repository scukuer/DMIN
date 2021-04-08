# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 26 Mar, 2021
Author : chenlangscu@163.com
prefer from https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/layers/interaction.py
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from deepctr.layers.normalization import LayerNormalization
from tensorflow.keras.initializers import TruncatedNormal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.dot_attention import DotAttention
from layers.softmax_weight_sum import SoftmaxWeightedSum


class SelfMultiHeadAttention_Split(Layer):
    """
      :param query: A 3d tensor with shape of [batch_size, T, C]
      :param key_masks: A 3d tensor with shape of [batch_size, 1]
      :return: A 3d tensor with shape of  [batch_size, head, T, C/head]
    """

    def __init__(self, num_units=8, head_num=4, scale=True, dropout_rate=0.2, future_binding=True, use_layer_norm=True,
                 use_res=True, seed=2020, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.num_units = num_units
        self.head_num = head_num
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.future_binding = future_binding
        self.use_layer_norm = use_layer_norm
        self.use_res = use_res
        self.seed = seed
        super(SelfMultiHeadAttention_Split, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `SelfMultiHeadAttention` layer should be called on a list of 2 tensors')
        if len(input_shape[0]) != 3 or len(input_shape[1]) != 2:
            raise ValueError('input: [N, T_k, d_model], key masks: [N, key_seqlen]')
        embedding_size = int(input_shape[0][-1])
        if self.num_units is None:
            self.num_units = embedding_size
        self.W = self.add_weight(name='Q_K_V', shape=[embedding_size, self.num_units * 3], dtype=tf.float32,
                                 initializer=TruncatedNormal(seed=self.seed))
        self.W_output = self.add_weight(name='output_W', shape=[self.num_units, self.num_units], dtype=tf.float32,
                                        initializer=TruncatedNormal(seed=self.seed))

        self.layer_norm = LayerNormalization()
        self.attention = DotAttention(scale=self.scale)
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=self.dropout_rate, future_binding=self.future_binding,
                                                     seed=self.seed)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)
        self.seq_len_max = int(input_shape[0][1])
        # Be sure to call this somewhere!
        super(SelfMultiHeadAttention_Split, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        input_info, keys_length = inputs
        hist_len = input_info.get_shape()[1]
        # tf.sequence_mask([[1],[2],[6]],5)
        # < tf.Tensor: shape = (3, 1, 5), dtype = bool, numpy =
        # array([[[True, False, False, False, False]],
        #        [[True, True, False, False, False]],
        #        [[True, True, True, True, True]]]) >
        key_masks = tf.sequence_mask(keys_length, hist_len)
        key_masks = tf.squeeze(key_masks, axis=1)  # [batch_size, T]

        # input_info: [batch_size, T, embedding_size]   W: [embedding_size, num_units*3]
        Q_K_V = tf.tensordot(input_info, self.W, axes=(-1, 0))  # [N T_q D*3]
        querys, keys, values = tf.split(Q_K_V, 3, -1)  # (batch, 5, 64) (batch, 5, 64) (batch, 5, 64)

        # head_num None F D
        # (h*batch, 5, 16) (h*batch, 5, 16) (h*batch, 5, 16)
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # (h*N, T_q, C/h)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)      # (h*N, T_k, C/h)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)  # (h*N, T_k, C/h)

        # (h*N, T_q, T_k)
        align = self.attention([querys, keys])

        key_masks = tf.tile(key_masks, [self.head_num, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input_info)[1], 1])  # (h*N, T_q, T_k)

        outputs = self.softmax_weight_sum([align, values, key_masks])  # (h*N, T_q, C/h)
        # (N, head, T_q, C/h)
        outputs = tf.transpose(tf.stack(tf.split(outputs, self.head_num, axis=0), axis=0), perm=[1, 0, 2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return None, self.head_num, input_shape[0][1], self.num_units/self.head_num

    def get_config(self, ):
        config = {'num_units': self.num_units, 'head_num': self.head_num, 'scale': self.scale,
                  'dropout_rate': self.dropout_rate,
                  'future_binding': self.future_binding, 'use_layer_norm': self.use_layer_norm, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(SelfMultiHeadAttention_Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask
