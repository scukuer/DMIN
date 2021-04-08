# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 1 April, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)


class EmbeddingIndex(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))