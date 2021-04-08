# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 08 Apr, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from tensorflow.python.keras.regularizers import l2
from layers.behavior_refiner_layer import Behavior_Refiner_Layer
from layers.multi_interest_extractor_layer import Multi_Interest_Extractor_Layer

class DMIN(tf.keras.Model):
    """
    Deep Multi-Interest Network
    """
    def __init__(self, input_vocab_size, d_model, num_head, position_vocab_size, position_dim, num_sampled=10, seq_mask_zero=True, l2_reg_embedding=1e-6,):
        super(DMIN, self).__init__(self)
        self.input_vocab_size = input_vocab_size
        self.d_model = d_model
        self.num_head = num_head
        self.num_sampled = num_sampled
        self.seq_mask_zero = seq_mask_zero
        self.l2_reg_embedding = l2_reg_embedding

        self.position_embedding = tf.keras.layers.Embedding(position_vocab_size, position_dim,
                                                            mask_zero=seq_mask_zero,
                                                            embeddings_regularizer=l2(l2_reg_embedding))

        self.brl = Behavior_Refiner_Layer(input_vocab_size, d_model, num_head)
        self.miel = Multi_Interest_Extractor_Layer(position_vocab_size, position_dim, d_model, num_head)

        hide1_layer = tf.keras.layers.Dense(units=128, activation='relu')
        hide2_layer = tf.keras.layers.Dense(units=64, activation='relu')
        hide3_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')
        self.seq_model = tf.keras.Sequential(layers=[hide1_layer, hide2_layer, hide3_layer])

    def call(self, inputs, training=None, mask=None):
        # input_id.shape:  [batch, T]
        # position.shape:  [batch, T]
        # target_id.shape: [batch, 1]
        # keys_length:     [batch, 1]
        input_id, position, target_id, keys_length = inputs

        brl_output = self.brl([input_id, keys_length])          # [batch, T, d_model]

        # 共用Behavior_Refiner_Layer中的self.embedding
        target_emb = self.brl.get_target_embedding(target_id)   # [batch, 1, d_model]

        # 用户多兴趣
        miel_output = self.miel([brl_output, keys_length, position, target_emb])   # [batch, head, d_model/head]

        miel_output = tf.reshape(miel_output, shape=[miel_output.get_shape()[0], -1])
        target_emb = tf.squeeze(target_emb)

        output = tf.concat([miel_output, target_emb], axis=-1)
        output = self.seq_model(output)    # [batch, 1]
        return output

    def get_aux_loss(self):
        return self.brl.get_aux_loss()


def debug():
    his_length = 8
    batch_size = 32
    input_vocab_size = 100
    d_model = 36
    num_head = 3
    position_vocab_size = 100
    position_dim = 20
    assert d_model % num_head == 0
    input_id = tf.convert_to_tensor(np.random.randint(low=0, high=input_vocab_size, size=(batch_size, his_length)))
    keys_length = tf.convert_to_tensor(np.random.randint(low=1, high=his_length + 1, size=(batch_size, 1)))
    target_id = tf.convert_to_tensor(np.random.randint(low=1, high=his_length + 1, size=(batch_size, 1)))
    position = tf.tile(tf.expand_dims(np.array(list(range(his_length))), axis=0), [batch_size, 1])

    dmin = DMIN(input_vocab_size, d_model, num_head, position_vocab_size,position_dim)
    output = dmin([input_id, position, target_id, keys_length])
    print("output: ", output)
    print("output shape: ", output.shape)     # (32, 1)


if __name__ == "__main__":
    print(tf.__version__)  # 2.3.0
    debug()

