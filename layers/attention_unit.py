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

from utils.Dice import *


class Attention_Unit(tf.keras.layers.Layer):
    """
    Attention_Unit
    """
    def __init__(self, d_model, head):
        super(Attention_Unit, self).__init__()
        assert d_model % head == 0
        self.d_model = d_model
        self.head = head

    def call(self, inputs, **kwargs):
        """
        :param inputs:
        :param kwargs:
        :return:
        """
        # input_embed.shape:      [batch, head, T, d_model/head]
        # position_embed.shape:   [batch, T, pos_embed]
        # target_embed.shape:     [batch, 1, d_model]
        input_embed, position_embed, target_embed = inputs
        head_num = input_embed.get_shape()[1]
        seq_len = input_embed.get_shape()[2]
        # [batch, head, T, pos_embed]
        position_embed = tf.tile(tf.expand_dims(position_embed, axis=1), multiples=[1, head_num, 1, 1])
        target_embed = target_embed[..., tf.newaxis, :]     # [batch, 1, 1, d_model]
        target_embed = tf.tile(target_embed, [1, head_num, seq_len, 1])  # [batch, head, T, d_model]
        target_position = tf.concat([position_embed, target_embed], axis=-1)  # [batch, head, T, pos_embed+d_model]
        target_position = tf.keras.layers.Dense(self.d_model/self.head, activation='tanh')(target_position)
        cross_product = target_position * input_embed  # [batch, head, T, d_model/head]
        concat = tf.concat([target_position, cross_product, input_embed], axis=-1)
        concat_dice = dice(concat)
        output = tf.keras.layers.Dense(1, activation='tanh')(concat_dice)
        output = tf.squeeze(output)
        return output


def debug():
    input_embed = tf.random.uniform((64, 4, 8, 4))
    position_embed = tf.random.uniform((64, 8, 10))
    target_embed = tf.random.uniform((64, 1, 16))
    au = Attention_Unit(16, 4)
    print("au: ", au([input_embed, position_embed, target_embed]))  # (64, 4, 8)


if __name__ == "__main__":
    print(tf.__version__)  # 2.3.0
    debug()





