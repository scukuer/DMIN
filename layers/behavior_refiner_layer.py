# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 01 Apr, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from tensorflow.python.keras.regularizers import l2
from utils.embedding_index import EmbeddingIndex
from tensorflow.keras.initializers import Zeros
from utils.position_embedding import positional_encoding
from layers.self_multi_head_attention import SelfMultiHeadAttention


class Behavior_Refiner_Layer(tf.keras.layers.Layer):
    """
    Behavior Refiner Layer
    """

    def __init__(self, input_vocab_size, d_model, num_head, num_sampled=10, seq_mask_zero=True, l2_reg_embedding=1e-6,
                 **kwargs):
        """
        :param input_vocab_size: 输入集合大小
        :param d_model: 向量维度
        :param kwargs:
        """
        super(Behavior_Refiner_Layer, self).__init__(**kwargs)
        assert d_model % num_head == 0
        self.input_vocab_size = input_vocab_size
        self.d_model = d_model
        self.aux_loss = 0.0
        self.num_sampled = num_sampled

        self.zero_bias = self.add_weight(shape=[input_vocab_size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model,
                                                   mask_zero=seq_mask_zero, embeddings_regularizer=l2(l2_reg_embedding))
        self.mha = SelfMultiHeadAttention(num_units=d_model, head_num=num_head)

    def call(self, inputs, **kwargs):
        # keys_length: [batch, 1]
        inputs_id, keys_length = inputs
        input_embedding = self.embedding(inputs_id)  # [batch, T, d_model]
        att_output = self.mha([input_embedding, keys_length])  # [batch, T, d_model]

        # aux loss
        self.aux_loss = self.compute_aux_loss(inputs_id, input_embedding, att_output, keys_length)

        return att_output

    def compute_aux_loss(self, inputs_id, input_embed, att_output, keys_length):
        input_id = inputs_id  # [batch, T]
        # input_embed = input_embed   # [batch, T, d_model]
        att_output = att_output  # [batch, T, d_model]
        keys_length = keys_length  # [batch, 1]

        item_index = EmbeddingIndex(list(range(self.input_vocab_size)))(0)  # [vocab_size,]
        item_emb = self.embedding(item_index)  # [vocab_size, dim]

        time_step = input_embed.get_shape()[1]  # T

        loss = []
        for step in range(time_step - 1):
            input_id_time = tf.expand_dims(input_id[:, step + 1], axis=-1)  # [batch,1]
            input_mask = tf.cast(tf.logical_not(tf.equal(input_id_time, 0)), dtype=tf.float32)  # [batch,1]
            # print("input_id_time: ", input_id_time)
            # print("input_mask: ", input_mask)
            output_time = att_output[:, step, :]  # [batch,d_model]

            # [batch,]
            loss_batch = tf.nn.sampled_softmax_loss(weights=item_emb,  # 全体的item embedding
                                                    biases=self.zero_bias,
                                                    labels=input_id_time,  # 传入的目标 batch item id
                                                    inputs=output_time,  # 模型的中间输入
                                                    num_sampled=self.num_sampled,
                                                    num_classes=self.input_vocab_size,  # item的词典大小
                                                    )
            # 填充的不参与计算
            loss_e = tf.reduce_sum(tf.expand_dims(loss_batch, axis=-1) * input_mask) / tf.reduce_sum(input_mask)

            loss.append(loss_e)

        return tf.reduce_mean(loss)

    def get_aux_loss(self):
        return self.aux_loss

    def get_target_embedding(self, target_id):
        target_emb = self.embedding(target_id)
        return target_emb   # [batch, 1, d_model]


def debug():
    his_length = 8
    batch_size = 32
    input_vocab_size = 100
    d_model = 36
    num_head = 6
    assert d_model % num_head == 0
    items = tf.convert_to_tensor(np.random.randint(low=0, high=input_vocab_size, size=(batch_size, his_length)))
    items_emb = tf.random.uniform((batch_size, his_length, d_model))
    keys_length = tf.convert_to_tensor(np.random.randint(low=1, high=his_length + 1, size=(batch_size, 1)))
    print("items shape: ", items.shape)  # (32, 8)
    print("keys_length shape: ", keys_length.shape)  # (32, 1)
    brl = Behavior_Refiner_Layer(input_vocab_size, d_model, num_head)
    output = brl([items, keys_length])

    print("output shape: ", output.shape)  # (32, 8, 36)

    print("aux_loss: ", brl.get_aux_loss())


if __name__ == "__main__":
    print(tf.__version__)  # 2.3.0
    debug()
