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
Image encoder classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 \
  import inception_v3_base

from seq2seq.encoders.encoder import Encoder, EncoderOutput


class InceptionV3Encoder(Encoder):
  """
  A unidirectional RNN encoder. Stacking should be performed as
  part of the cell.

  Params:
    resize_height: Resize the image to this height before feeding it
      into the convolutional network.
    resize_width: Resize the image to this width before feeding it
      into the convolutional network.
  """

  def __init__(self, params, mode, name="image_encoder"):
    super(InceptionV3Encoder, self).__init__(params, mode, name)

  @staticmethod
  def default_params():
    return {
        "resize_height": 299,
        "resize_width": 299,
    }

  def encode(self, inputs):
    inputs = tf.image.resize_images(
        images=inputs,
        size=[self.params["resize_height"], self.params["resize_width"]],
        method=tf.image.ResizeMethod.BILINEAR)

    outputs, _ = inception_v3_base(tf.to_float(inputs))
    output_shape = outputs.get_shape()  #pylint: disable=E1101
    shape_list = output_shape.as_list()

    # Take attentin over output elemnts in width and height dimension:
    # Shape: [B, W*H, ...]
    outputs_flat = tf.reshape(outputs, [shape_list[0], -1, shape_list[-1]])

    # Final state is the pooled output
    # Shape: [B, W*H*...]
    final_state = tf.contrib.slim.avg_pool2d(
        outputs, output_shape[1:3], padding="VALID", scope="pool")
    final_state = tf.contrib.slim.flatten(outputs, scope="flatten")

    return EncoderOutput(
        outputs=outputs_flat,
        final_state=final_state,
        attention_values=outputs_flat,
        attention_values_length=tf.shape(outputs_flat)[1])
