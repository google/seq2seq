"""
Test Cases for decoders.
"""

import tensorflow as tf
import numpy as np

from seq2seq.decoders import BasicDecoder, AttentionDecoder, AttentionLayer
from seq2seq.decoders import FixedDecoderInputs, DynamicDecoderInputs


class DecoderTests(object):
  """
  A collection of decoder tests. This class should be inherited together with
  `tf.test.TestCase`.
  """

  def __init__(self):
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.cell = tf.nn.rnn_cell.LSTMCell(32)
    self.vocab_size = 100
    self.max_decode_length = 16

  def create_decoder(self):
    """Creates the decoder module.

    This must be implemented by child classes and instantiate the appropriate
    decoder to be tested.
    """
    raise NotImplementedError

  def test_with_fixed_inputs(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    seq_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

    decoder_input_fn = FixedDecoderInputs(inputs, seq_length)
    decoder_fn = self.create_decoder()
    decoder_output, _, _ = decoder_fn(decoder_input_fn, initial_state,
                                      seq_length)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [self.batch_size, self.sequence_length, self.vocab_size])
    np.testing.assert_array_equal(decoder_output_.predictions.shape,
                                  [self.batch_size, self.sequence_length])

    return decoder_output_

  def test_gradients(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    seq_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    labels = np.random.randint(0, self.vocab_size,
                               [self.batch_size, self.sequence_length])

    decoder_input_fn = FixedDecoderInputs(inputs, seq_length)
    decoder_fn = self.create_decoder()
    decoder_output, _, _ = decoder_fn(decoder_input_fn, initial_state,
                                      seq_length)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        decoder_output.logits, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(tf.reduce_mean(losses))

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      grads_and_vars_ = sess.run(grads_and_vars)

    for grad, _ in grads_and_vars_:
      self.assertFalse(np.isnan(grad).any())

    return grads_and_vars_

  def test_with_dynamic_inputs(self):
    initial_input = tf.random_normal([self.batch_size, self.input_depth])
    seq_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    embeddings = tf.get_variable("W_embed", [self.vocab_size, self.input_depth])

    def make_input_fn(step_output):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, step_output.predictions)

    decoder_input_fn = DynamicDecoderInputs(initial_input, make_input_fn)
    decoder_fn = self.create_decoder()
    decoder_output, _, _ = decoder_fn(decoder_input_fn, initial_state,
                                      seq_length)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [self.batch_size, self.sequence_length, self.vocab_size])
    np.testing.assert_array_equal(decoder_output_.predictions.shape,
                                  [self.batch_size, self.sequence_length])

    return decoder_output


class BasicDecoderTest(tf.test.TestCase, DecoderTests):
  """Tests the `BasicDecoder` class.
  """

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.logging.set_verbosity(tf.logging.INFO)
    DecoderTests.__init__(self)

  def create_decoder(self):
    return BasicDecoder(
        cell=self.cell,
        vocab_size=self.vocab_size,
        max_decode_length=self.max_decode_length)


class AttentionDecoderTest(tf.test.TestCase, DecoderTests):
  """Tests the `AttentionDecoder` class.
  """

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.logging.set_verbosity(tf.logging.INFO)
    DecoderTests.__init__(self)
    self.attention_dim = 64
    self.input_seq_len = 10
    self.attention_inputs = tf.convert_to_tensor(
        np.random.randn(self.batch_size, self.input_seq_len, 32),
        dtype=tf.float32)

  def create_decoder(self):
    attention_fn = AttentionLayer(self.attention_dim)
    return AttentionDecoder(
        cell=self.cell,
        vocab_size=self.vocab_size,
        attention_inputs=self.attention_inputs,
        attention_fn=attention_fn,
        max_decode_length=self.max_decode_length)

  def test_attention_scores(self):
    decoder_output_ = self.test_with_fixed_inputs()
    np.testing.assert_array_equal(
        decoder_output_.attention_scores.shape,
        [self.batch_size, self.sequence_length, self.input_seq_len])

    # Make sure the attention scores sum to 1 for each step
    scores_sum = np.sum(decoder_output_.attention_scores, axis=2)
    np.testing.assert_array_almost_equal(
        scores_sum, np.ones([self.batch_size, self.sequence_length]))


if __name__ == "__main__":
  tf.test.main()
