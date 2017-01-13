"""
Test Cases for decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from seq2seq.decoders import BasicDecoder, AttentionDecoder, AttentionLayer
from seq2seq.decoders import beam_search_decoder
from seq2seq.decoders import FixedDecoderInputs, DynamicDecoderInputs
from seq2seq.inference import beam_search


class DecoderTests(object):
  """
  A collection of decoder tests. This class should be inherited together with
  `tf.test.TestCase`.
  """

  def __init__(self):
    self.batch_size = 4
    self.sequence_length = 16
    self.input_depth = 10
    self.cell = tf.contrib.rnn.LSTMCell(32)
    self.vocab_size = 100
    self.max_decode_length = 20

  def create_decoder(self, input_fn):
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
    decoder_fn = self.create_decoder(input_fn=decoder_input_fn)
    decoder_output, _, _ = decoder_fn(initial_state)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [self.sequence_length, self.batch_size, self.vocab_size])
    np.testing.assert_array_equal(decoder_output_.predicted_ids.shape,
                                  [self.sequence_length, self.batch_size])

    return decoder_output_

  def test_gradients(self):
    inputs = tf.random_normal(
        [self.batch_size, self.sequence_length, self.input_depth])
    seq_length = tf.ones(self.batch_size, dtype=tf.int32) * self.sequence_length
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    labels = np.random.randint(0, self.vocab_size,
                               [self.batch_size, self.sequence_length])

    decoder_input_fn = FixedDecoderInputs(inputs, seq_length)
    decoder_fn = self.create_decoder(input_fn=decoder_input_fn)
    decoder_output, _, _ = decoder_fn(initial_state)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=decoder_output.logits,
        labels=labels)
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
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    embeddings = tf.get_variable("W_embed", [self.vocab_size, self.input_depth])

    def make_input_fn(predicted_ids):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, predicted_ids)

    decoder_input_fn = DynamicDecoderInputs(
        initial_inputs=initial_input,
        make_input_fn=make_input_fn,
        max_decode_length=self.max_decode_length)
    decoder_fn = self.create_decoder(input_fn=decoder_input_fn)
    decoder_output, _, _ = decoder_fn(initial_state)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [self.max_decode_length, self.batch_size, self.vocab_size])
    np.testing.assert_array_equal(decoder_output_.predicted_ids.shape,
                                  [self.max_decode_length, self.batch_size])

  def test_inference_early_stopping(self):
    initial_input = tf.random_normal([self.batch_size, self.input_depth])
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    embeddings = tf.get_variable("W_embed", [self.vocab_size, self.input_depth])

    def make_input_fn(predicted_ids):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, predicted_ids)

    def elements_finished_fn(time_, predicted_ids):
      """Looks up the predictions in the embeddings.
      """
      ones_batch = tf.ones(tf.shape(predicted_ids[0]), dtype=time_.dtype)
      return (ones_batch * time_) >= 5

    decoder_input_fn = DynamicDecoderInputs(
        initial_inputs=initial_input,
        make_input_fn=make_input_fn,
        max_decode_length=self.max_decode_length,
        elements_finished_fn=elements_finished_fn)
    decoder_fn = self.create_decoder(input_fn=decoder_input_fn)
    decoder_output, _, _ = decoder_fn(initial_state)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [5, self.batch_size, self.vocab_size])
    np.testing.assert_array_equal(decoder_output_.predicted_ids.shape,
                                  [5, self.batch_size])

  def test_with_beam_search(self):
    # Batch size for beam search must be 1.
    self.batch_size = 1
    config = beam_search.BeamSearchConfig(
        beam_width=10,
        vocab_size=self.vocab_size,
        eos_token=self.vocab_size - 2,
        score_fn=beam_search.logprob_score,
        choose_successors_fn=beam_search.choose_top_k)

    initial_input = tf.random_normal([self.batch_size, self.input_depth])
    initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
    embeddings = tf.get_variable("W_embed", [self.vocab_size, self.input_depth])

    def make_input_fn(predicted_ids):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, predicted_ids)

    decoder_input_fn = DynamicDecoderInputs(
        initial_inputs=initial_input,
        make_input_fn=make_input_fn,
        max_decode_length=self.max_decode_length)
    decoder_fn = self.create_decoder(input_fn=decoder_input_fn)
    decoder_fn = beam_search_decoder.BeamSearchDecoder(
        decoder=decoder_fn, config=config)

    decoder_output, _, _ = decoder_fn(initial_state)

    #pylint: disable=E1101
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    np.testing.assert_array_equal(
        decoder_output_.logits.shape,
        [self.max_decode_length, 1, config.beam_width, self.vocab_size])
    np.testing.assert_array_equal(
        decoder_output_.predicted_ids.shape,
        [self.max_decode_length, 1, config.beam_width])
    np.testing.assert_array_equal(
        decoder_output_.beam_parent_ids.shape,
        [self.max_decode_length, 1, config.beam_width])
    np.testing.assert_array_equal(
        decoder_output_.scores.shape,
        [self.max_decode_length, 1, config.beam_width])
    np.testing.assert_array_equal(
        decoder_output_.original_outputs.predicted_ids.shape,
        [self.max_decode_length, 1, config.beam_width])
    np.testing.assert_array_equal(
        decoder_output_.original_outputs.logits.shape,
        [self.max_decode_length, 1, config.beam_width, self.vocab_size])

    return decoder_output


class BasicDecoderTest(tf.test.TestCase, DecoderTests):
  """Tests the `BasicDecoder` class.
  """

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.logging.set_verbosity(tf.logging.INFO)
    DecoderTests.__init__(self)

  def create_decoder(self, input_fn):
    return BasicDecoder(
        cell=self.cell,
        input_fn=input_fn,
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

  def create_decoder(self, input_fn):
    attention_fn = AttentionLayer(self.attention_dim)
    attention_inputs = tf.convert_to_tensor(
        np.random.randn(self.batch_size, self.input_seq_len, 32),
        dtype=tf.float32)
    return AttentionDecoder(
        cell=self.cell,
        input_fn=input_fn,
        vocab_size=self.vocab_size,
        attention_inputs=attention_inputs,
        attention_fn=attention_fn,
        max_decode_length=self.max_decode_length)

  def test_attention_scores(self):
    decoder_output_ = self.test_with_fixed_inputs()
    np.testing.assert_array_equal(
        decoder_output_.attention_scores.shape,
        [self.sequence_length, self.batch_size, self.input_seq_len])

    # Make sure the attention scores sum to 1 for each step
    scores_sum = np.sum(decoder_output_.attention_scores, axis=2)
    np.testing.assert_array_almost_equal(
        scores_sum, np.ones([self.sequence_length, self.batch_size]))


if __name__ == "__main__":
  tf.test.main()
