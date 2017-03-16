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
Test Cases for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from seq2seq.encoders import rnn_encoder


class UnidirectionalRNNEncoderTest(tf.test.TestCase):
  """
  Tests the UnidirectionalRNNEncoder class.
  """

  def setUp(self):
    super(UnidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN
    self.params = rnn_encoder.UnidirectionalRNNEncoder.default_params()
    self.params["rnn_cell"]["cell_params"]["num_units"] = 32
    self.params["rnn_cell"]["cell_class"] = "BasicLSTMCell"

  def test_encode(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.UnidirectionalRNNEncoder(self.params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    np.testing.assert_array_equal(encoder_output_.outputs.shape,
                                  [self.batch_size, self.sequence_length, 32])
    self.assertIsInstance(encoder_output_.final_state,
                          tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state.h.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state.c.shape,
                                  [self.batch_size, 32])


class BidirectionalRNNEncoderTest(tf.test.TestCase):
  """
  Tests the BidirectionalRNNEncoder class.
  """

  def setUp(self):
    super(BidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.params = rnn_encoder.BidirectionalRNNEncoder.default_params()
    self.params["rnn_cell"]["cell_params"]["num_units"] = 32
    self.params["rnn_cell"]["cell_class"] = "BasicLSTMCell"
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def test_encode(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.BidirectionalRNNEncoder(self.params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, 32 * 2])

    self.assertIsInstance(encoder_output_.final_state[0],
                          tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1],
                          tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state[0].h.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[0].c.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[1].h.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[1].c.shape,
                                  [self.batch_size, 32])


class StackBidirectionalRNNEncoderTest(tf.test.TestCase):
  """
  Tests the StackBidirectionalRNNEncoder class.
  """

  def setUp(self):
    super(StackBidirectionalRNNEncoderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.mode = tf.contrib.learn.ModeKeys.TRAIN

  def _test_encode_with_params(self, params):
    """Tests the StackBidirectionalRNNEncoder with a specific cell"""
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.StackBidirectionalRNNEncoder(params, self.mode)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    output_size = encode_fn.params["rnn_cell"]["cell_params"]["num_units"]

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, output_size * 2])

    return encoder_output_

  def test_encode_with_single_cell(self):
    encoder_output_ = self._test_encode_with_params({
        "rnn_cell": {
            "num_layers": 1,
            "cell_params": {
                "num_units": 32
            }
        }
    })

    self.assertIsInstance(encoder_output_.final_state[0][0],
                          tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1][0],
                          tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state[0][0].h.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[0][0].c.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[1][0].h.shape,
                                  [self.batch_size, 32])
    np.testing.assert_array_equal(encoder_output_.final_state[1][0].c.shape,
                                  [self.batch_size, 32])

  def test_encode_with_multi_cell(self):
    encoder_output_ = self._test_encode_with_params({
        "rnn_cell": {
            "num_layers": 4,
            "cell_params": {
                "num_units": 32
            }
        }
    })

    for layer_idx in range(4):
      self.assertIsInstance(encoder_output_.final_state[0][layer_idx],
                            tf.contrib.rnn.LSTMStateTuple)
      self.assertIsInstance(encoder_output_.final_state[1][layer_idx],
                            tf.contrib.rnn.LSTMStateTuple)
      np.testing.assert_array_equal(
          encoder_output_.final_state[0][layer_idx].h.shape,
          [self.batch_size, 32])
      np.testing.assert_array_equal(
          encoder_output_.final_state[0][layer_idx].c.shape,
          [self.batch_size, 32])
      np.testing.assert_array_equal(
          encoder_output_.final_state[1][layer_idx].h.shape,
          [self.batch_size, 32])
      np.testing.assert_array_equal(
          encoder_output_.final_state[1][layer_idx].c.shape,
          [self.batch_size, 32])


if __name__ == "__main__":
  tf.test.main()
