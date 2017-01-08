"""
Test Cases for Training utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.data import data_utils
from seq2seq.test import utils as test_utils
from seq2seq.training import utils as training_utils
from seq2seq.training import HParamsParser, hooks


class TestHparamsReadWrite(tf.test.TestCase):
  """Tests reading and writing of HParam values
  """

  def setUp(self):
    super(TestHparamsReadWrite, self).setUp()
    self.hparams = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    self.parser = HParamsParser(self.hparams)

  def test_write(self):
    file = tempfile.NamedTemporaryFile()
    training_utils.write_hparams(self.hparams, file.name)
    lines = file.read()
    self.assertEqual(
        lines.decode(),
        "dropout=0.8\nnum_layers=2\nrnn_cell_type=LSTM\nrnn_dim=128")
    file.close()

  def test_write_read(self):
    file = tempfile.NamedTemporaryFile()
    training_utils.write_hparams(self.hparams, file.name)
    params_str = training_utils.read_hparams(file.name)
    final_params = self.parser.parse(params_str)
    self.assertEqual(final_params, self.hparams)


class TestInputFn(tf.test.TestCase):
  """Tests create_input_fn"""
  def _test_with_args(self, **kwargs):
    """Helper function to test create_input_fn with keyword arguments"""
    sources_file, targets_file = test_utils.create_temp_parallel_data(
        sources=["Hello World ."],
        targets=["Goodbye ."]
    )
    data_provider_fn = lambda: data_utils.make_parallel_data_provider(
        [sources_file.name], [targets_file.name])
    input_fn = training_utils.create_input_fn(
        data_provider_fn=data_provider_fn,
        **kwargs)
    features, labels = input_fn()

    with self.test_session() as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        features_, labels_ = sess.run([features, labels])

    self.assertEqual(features_.keys(), set(["source_tokens", "source_len"]))
    self.assertEqual(labels_.keys(), set(["target_tokens", "target_len"]))

  def test_without_buckets(self):
    self._test_with_args(batch_size=10)

  def test_wit_buckets(self):
    self._test_with_args(batch_size=10, bucket_boundaries=[0, 5, 10])


class TestDefaultHooks(tf.test.TestCase):
  """Tests the creation of default training hooks"""
  def test_hooks(self):
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=tf.contrib.layers.sparse_column_with_hash_bucket(
            "test", 10))
    default_hooks = training_utils.create_default_training_hooks(
        estimator=estimator)

    hook_types = list(map(type, default_hooks))
    self.assertIn(hooks.PrintModelAnalysisHook, hook_types)
    self.assertIn(hooks.TrainSampleHook, hook_types)
    self.assertIn(hooks.MetadataCaptureHook, hook_types)
    self.assertIn(hooks.TokensPerSecondCounter, hook_types)


class TestMosesBleu(tf.test.TestCase):
  """Tests using the Moses multi-bleu script to calcualte BLEU score"""
  def test_multi_bleu(self):
    result = training_utils.moses_multi_bleu(
        hypotheses=np.array(["The brown fox jumps over the dog"]),
        references=np.array(["The quick brown fox jumps over the lazy dog"]),
        lowercase=False)
    self.assertEqual(result, 50.25)

  def test_multi_bleu_lowercase(self):
    result = training_utils.moses_multi_bleu(
        hypotheses=np.array(["The brown fox jumps over The Dog"]),
        references=np.array(["The quick brown fox jumps over the lazy dog"]),
        lowercase=True)
    self.assertEqual(result, 50.25)

  def test_multi_bleu_with_eos(self):
    result = training_utils.moses_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over the dog SEQUENCE_END2 x x x"]),
        references=np.array(["The quick brown fox jumps over the lazy dog"]),
        lowercase=False)
    self.assertEqual(result, 50.25)


class TestLRDecay(tf.test.TestCase):
  """Tests learning rate decay function.
  """

  def test_no_decay(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type=None, decay_steps=5, decay_rate=2.0)
    self.assertEqual(decay_fn, None)

    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="", decay_steps=5, decay_rate=2.0)
    self.assertEqual(decay_fn, None)

  def test_decay_without_min(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="exponential_decay",
        decay_steps=10,
        decay_rate=0.9,
        start_decay_at=100,
        stop_decay_at=1000,
        staircase=False)

    initial_lr = 1.0
    with self.test_session() as sess:
      # Should not decay before start_decay_at
      np.testing.assert_equal(sess.run(decay_fn(initial_lr, 50)), initial_lr)
      # Proper decay
      np.testing.assert_almost_equal(
          sess.run(decay_fn(initial_lr, 115)), initial_lr * 0.9**(15.0 / 10.0))
      # Should not decay past stop_decay_at
      np.testing.assert_almost_equal(
          sess.run(decay_fn(initial_lr, 5000)), initial_lr * 0.9**(
              (1000.0 - 100.0) / 10.0))

  def test_decay_with_min(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="exponential_decay",
        decay_steps=10,
        decay_rate=0.9,
        start_decay_at=100,
        stop_decay_at=1000,
        min_learning_rate=0.01,
        staircase=False)

    initial_lr = 1.0
    with self.test_session() as sess:
      # Should not decay past min_learning_rate
      np.testing.assert_almost_equal(sess.run(decay_fn(initial_lr, 900)), 0.01)


if __name__ == '__main__':
  tf.test.main()
