"""
Tests for Models
"""

from collections import namedtuple

from seq2seq import inputs
from seq2seq import losses as seq2seq_losses
from seq2seq.training import utils as training_utils
from seq2seq.test import utils as test_utils
from seq2seq.models import BasicSeq2Seq, AttentionSeq2Seq
from seq2seq.decoders import FixedDecoderInputs, DynamicDecoderInputs

import tensorflow as tf
import numpy as np


class EncoderDecoderTests(tf.test.TestCase):
  """Base class for EncoderDecoder tests. Tests for specific classes should
  inherit from this and tf.test.TestCase.
  """

  def setUp(self):
    super(EncoderDecoderTests, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.input_depth = 32
    self.max_decode_length = 40

    # Create vocabulary
    self.vocab_size = 100
    self.vocab_list = [str(_) for _ in range(self.vocab_size)]
    self.vocab_file = test_utils.create_temporary_vocab_file(self.vocab_list)
    self.vocab_info = inputs.get_vocab_info(self.vocab_file.name)

  def tearDown(self):
    self.vocab_file.close()

  def create_model(self):
    """Creates model class to be tested. Subclasses must implement this method.
    """
    self.skipTest("Base module should not be tested.")

  def _create_example(self):
    """Creates example data for a test"""
    source = np.random.randn(self.batch_size, self.max_decode_length,
                             self.input_depth)
    source_len = np.random.randint(0, self.max_decode_length, [self.batch_size])
    target_len = np.random.randint(0, self.max_decode_length * 2,
                                   [self.batch_size])
    target = np.random.randn(self.batch_size,
                             np.max(target_len), self.input_depth)
    labels = np.random.randint(0, self.vocab_size,
                               [self.batch_size, np.max(target_len) - 1])

    example_ = namedtuple(
        "Example", ["source", "source_len", "target", "target_len", "labels"])
    return example_(source, source_len, target, target_len, labels)

  def test_forward_pass(self):
    """Tests model forward pass by checking the shape of the outputs."""
    ex = self._create_example()
    decoder_input_fn = FixedDecoderInputs(
        inputs=tf.convert_to_tensor(
            ex.target, dtype=tf.float32),
        sequence_length=tf.convert_to_tensor(
            ex.target_len, dtype=tf.int32))

    model = self.create_model()
    decoder_output = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decoder_input_fn=decoder_input_fn,
        target_len=tf.convert_to_tensor(
            ex.target_len, dtype=tf.int32))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(ex.target_len, max_decode_length)

    # Assert shapes are correct
    np.testing.assert_array_equal(decoder_output_.logits.shape, [
        self.batch_size, np.max(expected_decode_len),
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(
        decoder_output_.predictions.shape,
        [self.batch_size, np.max(expected_decode_len)])

  def test_inference(self):
    """Tests model inference by feeding dynamic inputs based on an embedding
    """
    model = self.create_model()
    ex = self._create_example()

    embeddings = tf.get_variable(
        "W_embed", [model.target_vocab_info.total_size, self.input_depth])

    def make_input_fn(step_output):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, step_output.predictions)

    decoder_input_fn = DynamicDecoderInputs(
        initial_inputs=tf.zeros(
            [self.batch_size, self.input_depth], dtype=tf.float32),
        make_input_fn=make_input_fn)

    decoder_output = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decoder_input_fn=decoder_input_fn,
        target_len=self.max_decode_length)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    # Assert shapes are correct
    np.testing.assert_array_equal(decoder_output_.logits.shape, [
        self.batch_size, self.max_decode_length,
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(decoder_output_.predictions.shape,
                                  [self.batch_size, self.max_decode_length])

  def test_gradients(self):
    """Ensures the parameter gradients can be computed and are not NaN
    """
    ex = self._create_example()
    decoder_input_fn = FixedDecoderInputs(
        inputs=tf.convert_to_tensor(
            ex.target, dtype=tf.float32),
        sequence_length=tf.convert_to_tensor(
            ex.target_len, dtype=tf.int32))

    model = self.create_model()
    decoder_output = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decoder_input_fn=decoder_input_fn,
        target_len=tf.convert_to_tensor(
            ex.target_len, dtype=tf.int32))

    # Get a loss to optimize
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits,
        targets=tf.ones_like(decoder_output.predictions),
        sequence_length=tf.convert_to_tensor(
            ex.target_len, dtype=tf.int32))
    mean_loss = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(mean_loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, grads_and_vars_ = sess.run([train_op, grads_and_vars])

    for grad, _ in grads_and_vars_:
      self.assertFalse(np.isnan(grad).any())

  def test_pipeline(self):
    # Create source and target example
    source_len = 10
    target_len = self.max_decode_length + 10
    source = " ".join(np.random.choice(self.vocab_list, source_len))
    target = " ".join(np.random.choice(self.vocab_list, target_len))
    tfrecords_file = test_utils.create_temp_tfrecords(
        source=source, target=target)

    # Build model graph
    model = self.create_model()
    featurizer = model.create_featurizer()
    data_provider = lambda: inputs.make_data_provider([tfrecords_file.name])
    input_fn = training_utils.create_input_fn(data_provider, featurizer,
                                              self.batch_size)
    features, labels = input_fn()
    predictions, loss, train_op = model(features, labels, None,
                                        tf.contrib.learn.ModeKeys.TRAIN)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.initialize_all_tables())
      with tf.contrib.slim.queues.QueueRunners(sess):
        predictions_, loss_, _ = sess.run([predictions, loss, train_op])

    # We have predictions for each target words and the SEQUENCE_START token.
    # That's why it's `target_len + 1`
    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(target_len + 1, max_decode_length)

    np.testing.assert_array_equal(predictions_["logits"].shape, [
        self.batch_size, expected_decode_len, model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(predictions_["predictions"].shape,
                                  [self.batch_size, expected_decode_len])
    self.assertFalse(np.isnan(loss_))

    tfrecords_file.close()


class TestBasicSeq2Seq(EncoderDecoderTests):
  """Tests the seq2seq.models.BasicSeq2Seq model.
  """

  def setUp(self):
    super(TestBasicSeq2Seq, self).setUp()

  def create_model(self):
    return BasicSeq2Seq(
        source_vocab_info=self.vocab_info,
        target_vocab_info=self.vocab_info,
        params=BasicSeq2Seq.default_params())


class TestAttentionSeq2Seq(EncoderDecoderTests):
  """Tests the seq2seq.models.AttentionSeq2Seq model.
  """

  def setUp(self):
    super(TestAttentionSeq2Seq, self).setUp()
    self.encoder_rnn_cell = tf.nn.rnn_cell.LSTMCell(32)
    self.decoder_rnn_cell = tf.nn.rnn_cell.LSTMCell(32)
    self.attention_dim = 128

  def create_model(self):
    return AttentionSeq2Seq(
        source_vocab_info=self.vocab_info,
        target_vocab_info=self.vocab_info,
        params=AttentionSeq2Seq.default_params())


if __name__ == "__main__":
  tf.test.main()
