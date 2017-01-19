"""
Test Cases for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    self.cell = tf.contrib.rnn.LSTMCell(32)

  def test_encode(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.UnidirectionalRNNEncoder(self.cell)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, self.cell.output_size])
    self.assertIsInstance(encoder_output_.final_state,
                          tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state.h.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state.c.shape,
                                  [self.batch_size, self.cell.output_size])


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
    self.cell = tf.contrib.rnn.LSTMCell(32)

  def test_encode(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.BidirectionalRNNEncoder(self.cell)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, self.cell.output_size * 2])

    self.assertIsInstance(encoder_output_.final_state[0],
                          tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1],
                          tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state[0].h.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[0].c.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[1].h.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[1].c.shape,
                                  [self.batch_size, self.cell.output_size])


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

  def _test_encode_with_cell(self, cell):
    """Tests the StackBidirectionalRNNEncoder with a specific cell"""
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    example_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length

    encode_fn = rnn_encoder.StackBidirectionalRNNEncoder(cell)
    encoder_output = encode_fn(inputs, example_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoder_output_ = sess.run(encoder_output)

    np.testing.assert_array_equal(
        encoder_output_.outputs.shape,
        [self.batch_size, self.sequence_length, cell.output_size * 2])

    return encoder_output_

  def test_encode_with_single_cell(self):
    cell = tf.contrib.rnn.LSTMCell(32)
    encoder_output_ = self._test_encode_with_cell(cell)

    self.assertIsInstance(
        encoder_output_.final_state[0][0],
        tf.contrib.rnn.LSTMStateTuple)
    self.assertIsInstance(
        encoder_output_.final_state[1][0],
        tf.contrib.rnn.LSTMStateTuple)
    np.testing.assert_array_equal(
        encoder_output_.final_state[0][0].h.shape,
        [self.batch_size, cell.output_size])
    np.testing.assert_array_equal(
        encoder_output_.final_state[0][0].c.shape,
        [self.batch_size, cell.output_size])
    np.testing.assert_array_equal(
        encoder_output_.final_state[1][0].h.shape,
        [self.batch_size, cell.output_size])
    np.testing.assert_array_equal(
        encoder_output_.final_state[1][0].c.shape,
        [self.batch_size, cell.output_size])

  def test_encode_with_multi_cell(self):
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(32)] * 4)
    encoder_output_ = self._test_encode_with_cell(cell)

    for layer_idx in range(4):
      self.assertIsInstance(
          encoder_output_.final_state[0][layer_idx],
          tf.contrib.rnn.LSTMStateTuple)
      self.assertIsInstance(
          encoder_output_.final_state[1][layer_idx],
          tf.contrib.rnn.LSTMStateTuple)
      np.testing.assert_array_equal(
          encoder_output_.final_state[0][layer_idx].h.shape,
          [self.batch_size, cell.output_size])
      np.testing.assert_array_equal(
          encoder_output_.final_state[0][layer_idx].c.shape,
          [self.batch_size, cell.output_size])
      np.testing.assert_array_equal(
          encoder_output_.final_state[1][layer_idx].h.shape,
          [self.batch_size, cell.output_size])
      np.testing.assert_array_equal(
          encoder_output_.final_state[1][layer_idx].c.shape,
          [self.batch_size, cell.output_size])


if __name__ == "__main__":
  tf.test.main()
