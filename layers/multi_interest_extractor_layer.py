# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 07 Apr, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.attention_unit import Attention_Unit
from tensorflow.python.keras.regularizers import l2
from layers.self_multi_head_attention_split import SelfMultiHeadAttention_Split


class Multi_Interest_Extractor_Layer(tf.keras.layers.Layer):
    """
    Multi_Interest_Extractor_Layer
    """

    def __init__(self, position_vocab_size, position_dim, d_model, num_head, seq_mask_zero=True, l2_reg_embedding=1e-6):
        super(Multi_Interest_Extractor_Layer, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.position_embedding = tf.keras.layers.Embedding(position_vocab_size, position_dim,
                                                            mask_zero=seq_mask_zero,
                                                            embeddings_regularizer=l2(l2_reg_embedding))
        # 输出多个embedding
        self.mha = SelfMultiHeadAttention_Split(num_units=d_model, head_num=num_head)
        self.au = Attention_Unit(d_model, num_head)

    def call(self, inputs, **kwargs):
        # inputs_embed: [batch, T, d_model]
        # keys_length:  [batch, 1]
        # position:     [batch, T]
        # target_embed: [batch, 1, d_model]   -- 共用Behavior_Refiner_Layer中的self.embedding
        inputs_embed, keys_length, position, target_embed = inputs

        position_embed = self.position_embedding(position)               # [batch, T, pos_embed]
        att_output = self.mha([inputs_embed, keys_length])               # [batch, head, T, d_model/head]
        au_weight = self.au([att_output, position_embed, target_embed])  # [batch, head, T]

        # zero mask
        key_masks = tf.sequence_mask(keys_length, inputs_embed.get_shape()[1])
        key_masks = tf.tile(key_masks, [1, self.num_head, 1])     # [batch, head, T]
        paddings = tf.ones_like(au_weight) * (-2 ** 32 + 1)       # [batch, head, T]
        align = tf.where(key_masks, au_weight, paddings)          # [batch, head, T]
        align = tf.expand_dims(tf.nn.softmax(align), axis=-1)

        output = att_output * align             # [batch, head, T, d_model/head]
        output = tf.reduce_sum(output, axis=2)  # [batch, head, d_model/head]
        return output


def debug():
    T = 8
    batch = 64
    d_model = 16
    input_embed = tf.random.uniform((batch, T, d_model))
    target_embed = tf.random.uniform((batch, 1, d_model))
    keys_length = tf.convert_to_tensor(np.random.randint(low=1, high=T + 1, size=(batch, 1)))
    position = tf.tile(tf.expand_dims(np.array(list(range(T))), axis=0), [batch, 1])
    position_vocab_size = 100
    position_dim = 20
    num_head = 4
    assert d_model % num_head == 0
    miel = Multi_Interest_Extractor_Layer(position_vocab_size, position_dim, d_model, num_head)
    output = miel([input_embed, keys_length, position, target_embed])
    print("output shape: ", output.shape)    # (64, 4, 4)


if __name__ == "__main__":
    print(tf.__version__)  # 2.3.0
    debug()
