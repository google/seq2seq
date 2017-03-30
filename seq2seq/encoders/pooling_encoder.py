# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
An encoder that pools over embeddings, as described in
https://arxiv.org/abs/1611.02344.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydoc import locate

import numpy as np
import tensorflow as tf

from seq2seq.encoders.encoder import Encoder, EncoderOutput


def position_encoding(sentence_size, embedding_size):
  """
  Position Encoding described in section 4.1 of
  End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).

  Args:
    sentence_size: length of the sentence
    embedding_size: dimensionality of the embeddings

  Returns:
    A numpy array of shape [sentence_size, embedding_size] containing
    the fixed position encodings for each sentence position.
  """
  encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for k in range(1, le):
    for j in range(1, ls):
      encoding[j-1, k-1] = (1.0 - j/float(ls)) - (
          k / float(le)) * (1. - 2. * j/float(ls))
  return encoding


def _create_position_embedding(embedding_dim, num_positions, lengths, maxlen):
  """Creates position embeddings.

  Args:
    embedding_dim: Dimensionality of the embeddings. An integer.
    num_positions: The number of positions to be embedded. For example,
      if you have inputs of length up to 100, this should be 100. An integer.
    lengths: The lengths of the inputs to create position embeddings for.
      An int32 tensor of shape `[batch_size]`.
    maxlen: The maximum length of the input sequence to create position
      embeddings for. An int32 tensor.

  Returns:
    A tensor of shape `[batch_size, maxlen, embedding_dim]` that contains
    embeddings for each position. All elements past `lengths` are zero.
  """
  # Create constant position encodings
  position_encodings = tf.constant(
      position_encoding(num_positions, embedding_dim),
      name="position_encoding")

  # Slice to size of current sequence
  pe_slice = position_encodings[:maxlen, :]
  # Replicate encodings for each element in the batch
  batch_size = tf.shape(lengths)[0]
  pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

  # Mask out positions that are padded
  positions_mask = tf.sequence_mask(
      lengths=lengths, maxlen=maxlen, dtype=tf.float32)
  positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

  return positions_embed

class PoolingEncoder(Encoder):
  """An encoder that pools over embeddings, as described in
  https://arxiv.org/abs/1611.02344. The encoder supports optional positions
  embeddings and a configurable pooling window.

  Params:
    dropout_keep_prob: Dropout keep probability applied to the embeddings.
    pooling_fn: The 1-d pooling function to use, e.g.
      `tensorflow.layers.average_pooling1d`.
    pool_size: The pooling window, passed as `pool_size` to
      the pooling function.
    strides: The stride during pooling, passed as `strides`
      the pooling function.
    position_embeddings.enable: If true, add position embeddings to the
      inputs before pooling.
    position_embeddings.combiner_fn: Function used to combine the
      position embeddings with the inputs. For example, `tensorflow.add`.
    position_embeddings.num_positions: Size of the position embedding matrix.
      This should be set to the maximum sequence length of the inputs.
  """

  def __init__(self, params, mode, name="pooling_encoder"):
    super(PoolingEncoder, self).__init__(params, mode, name)
    self._pooling_fn = locate(self.params["pooling_fn"])
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

  @staticmethod
  def default_params():
    return {
        "dropout_keep_prob": 0.8,
        "pooling_fn": "tensorflow.layers.average_pooling1d",
        "pool_size": 5,
        "strides": 1,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.multiply",
        "position_embeddings.num_positions": 100,
    }

  def encode(self, inputs, sequence_length):
    if self.params["position_embeddings.enable"]:
      positions_embed = _create_position_embedding(
          embedding_dim=inputs.get_shape().as_list()[-1],
          num_positions=self.params["position_embeddings.num_positions"],
          lengths=sequence_length,
          maxlen=tf.shape(inputs)[1])
      inputs = self._combiner_fn(inputs, positions_embed)

    # Apply dropout
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=self.params["dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)

    outputs = self._pooling_fn(
        inputs=inputs,
        pool_size=self.params["pool_size"],
        strides=self.params["strides"],
        padding="SAME")

    # Final state is the average representation of the pooled embeddings
    final_state = tf.reduce_mean(outputs, 1)

    return EncoderOutput(
        outputs=outputs,
        final_state=final_state,
        attention_values=inputs,
        attention_values_length=sequence_length)
