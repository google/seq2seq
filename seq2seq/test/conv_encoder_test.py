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
Test Cases for PoolingEncoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from seq2seq.encoders import ConvEncoder


class ConvEncoderTest(tf.test.TestCase):
  """
  Tests the ConvEncoder class.
  """

  def setUp(self):
    super(ConvEncoderTest, self).setUp()
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def _test_with_params(self, params):
    """Tests the encoder with a given parameter configuration"""
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = ConvEncoder(params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    att_value_units = encode_fn.params["attention_cnn.units"]
    output_units = encode_fn.params["output_cnn.units"]

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, att_value_units])
    np.testing.assert_array_equal(
        encoder_output_.attention_values.shape,
        [self.batch_size, self.sequence_length, output_units])
    np.testing.assert_array_equal(
        encoder_output_.final_state.shape,
        [self.batch_size, output_units])

  def test_encode_with_pos(self):
    self._test_with_params({
        "position_embeddings.enable": True,
        "position_embeddings.num_positions": self.sequence_length,
        "attention_cnn.units": 5,
        "output_cnn.units": 6
    })

if __name__ == "__main__":
  tf.test.main()
