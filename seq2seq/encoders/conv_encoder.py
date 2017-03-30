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

import tensorflow as tf

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.encoders.pooling_encoder import _create_position_embedding


class ConvEncoder(Encoder):
  """A deep convolutional encoder, as described in
  https://arxiv.org/abs/1611.02344. The encoder supports optional positions
  embeddings.

  Params:
    attention_cnn.units: Number of units in `cnn_a`. Same in each layer.
    attention_cnn.kernel_size: Kernel size for `cnn_a`.
    attention_cnn.layers: Number of layers in `cnn_a`.
    embedding_dropout_keep_prob: Dropout keep probability
      applied to the embeddings.
    output_cnn.units: Number of units in `cnn_c`. Same in each layer.
    output_cnn.kernel_size: Kernel size for `cnn_c`.
    output_cnn.layers: Number of layers in `cnn_c`.
    position_embeddings.enable: If true, add position embeddings to the
      inputs before pooling.
    position_embeddings.combiner_fn: Function used to combine the
      position embeddings with the inputs. For example, `tensorflow.add`.
    position_embeddings.num_positions: Size of the position embedding matrix.
      This should be set to the maximum sequence length of the inputs.
  """

  def __init__(self, params, mode, name="conv_encoder"):
    super(ConvEncoder, self).__init__(params, mode, name)
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

  @staticmethod
  def default_params():
    return {
        "attention_cnn.units": 512,
        "attention_cnn.kernel_size": 3,
        "attention_cnn.layers": 15,
        "embedding_dropout_keep_prob": 0.8,
        "output_cnn.units": 256,
        "output_cnn.kernel_size": 3,
        "output_cnn.layers": 5,
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

    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)

    with tf.variable_scope("cnn_a"):
      cnn_a_output = inputs
      for layer_idx in range(self.params["attention_cnn.layers"]):
        next_layer = tf.contrib.layers.conv2d(
            inputs=cnn_a_output,
            num_outputs=self.params["attention_cnn.units"],
            kernel_size=self.params["attention_cnn.kernel_size"],
            padding="SAME",
            activation_fn=None)
        # Add a residual connection, except for the first layer
        if layer_idx > 0:
          next_layer += cnn_a_output
        cnn_a_output = tf.tanh(next_layer)

    with tf.variable_scope("cnn_c"):
      cnn_c_output = inputs
      for layer_idx in range(self.params["output_cnn.layers"]):
        next_layer = tf.contrib.layers.conv2d(
            inputs=cnn_c_output,
            num_outputs=self.params["output_cnn.units"],
            kernel_size=self.params["output_cnn.kernel_size"],
            padding="SAME",
            activation_fn=None)
        # Add a residual connection, except for the first layer
        if layer_idx > 0:
          next_layer += cnn_c_output
        cnn_c_output = tf.tanh(next_layer)

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return EncoderOutput(
        outputs=cnn_a_output,
        final_state=final_state,
        attention_values=cnn_c_output,
        attention_values_length=sequence_length)
