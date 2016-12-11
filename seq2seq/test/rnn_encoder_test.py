"""
Test Cases for RNN encoders.
"""

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
    self.cell = tf.nn.rnn_cell.LSTMCell(32)

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
                          tf.nn.rnn_cell.LSTMStateTuple)
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
    self.cell = tf.nn.rnn_cell.LSTMCell(32)

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
                          tf.nn.rnn_cell.LSTMStateTuple)
    self.assertIsInstance(encoder_output_.final_state[1],
                          tf.nn.rnn_cell.LSTMStateTuple)
    np.testing.assert_array_equal(encoder_output_.final_state[0].h.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[0].c.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[1].h.shape,
                                  [self.batch_size, self.cell.output_size])
    np.testing.assert_array_equal(encoder_output_.final_state[1].c.shape,
                                  [self.batch_size, self.cell.output_size])


if __name__ == "__main__":
  tf.test.main()
