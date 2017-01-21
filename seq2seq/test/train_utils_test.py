# -*- coding: utf-8 -*-

"""
Test Cases for Training utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import shutil
import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.contrib import rnn_cell
from seq2seq.data import data_utils
from seq2seq.test import utils as test_utils
from seq2seq.training import utils as training_utils
from seq2seq.training import hooks

class TestGetRNNCell(tf.test.TestCase):
  """Tests the get_rnn_cell function.
  """
  def test_single_layer(self):
    cell = training_utils.get_rnn_cell(
        cell_spec={"class": "BasicLSTMCell", "num_units": 16},
        num_layers=1)
    self.assertIsInstance(cell, tf.contrib.rnn.BasicLSTMCell)
    self.assertEqual(cell.output_size, 16)

  def test_multi_layer(self):
    cell = training_utils.get_rnn_cell(
        cell_spec={"class": "BasicLSTMCell", "num_units": 16},
        num_layers=2)
    self.assertIsInstance(cell, rnn_cell.ExtendedMultiRNNCell)
    self.assertEqual(cell.output_size, 16)

  def test_dropout(self):
    cell = training_utils.get_rnn_cell(
        cell_spec={"class": "BasicLSTMCell", "num_units": 16},
        num_layers=1,
        dropout_input_keep_prob=0.5)
    self.assertIsInstance(cell, tf.contrib.rnn.DropoutWrapper)
    self.assertEqual(cell.output_size, 16)

  def test_extra_args(self):

    # Invalid args should raise a ValueError
    with self.assertRaises(ValueError):
      training_utils.get_rnn_cell(
          cell_spec={"class": "LSTMCell", "num_units": 16,
                     "use_peepholesERROR": True},
          num_layers=1)

    cell = training_utils.get_rnn_cell(
        cell_spec={"class": "LSTMCell", "num_units": 16,
                   "use_peepholes": True, "forget_bias": 0.5},
        num_layers=1)
    self.assertIsInstance(cell, tf.contrib.rnn.LSTMCell)
    #pylint: disable=E1101,W0212
    self.assertEqual(cell._use_peepholes, True)
    self.assertEqual(cell._forget_bias, 0.5)
    self.assertEqual(cell.output_size, 16)


class TestTrainOptions(tf.test.TestCase):
  """Tests reading and writing of training options"""

  def setUp(self):
    super(TestTrainOptions, self).setUp()
    self.model_dir = tempfile.mkdtemp()
    self.hparams = {
        "num_layers": 4
    }
    self.model_class = "AttentionSeq2Seq"
    self.source_vocab_file = "some_file"
    self.target_vocab_file = "some_file2"

  def tearDown(self):
    super(TestTrainOptions, self).tearDown()
    shutil.rmtree(self.model_dir)

  def test_read_write(self):
    saved_opts = training_utils.TrainOptions(
        hparams=self.hparams,
        model_class=self.model_class,
        source_vocab_path=self.source_vocab_file,
        target_vocab_path=self.target_vocab_file)
    saved_opts.dump(self.model_dir)

    loaded_opt = training_utils.TrainOptions.load(
        model_dir=self.model_dir)

    self.assertEqual(saved_opts.hparams, loaded_opt.hparams)
    self.assertEqual(saved_opts.model_class, loaded_opt.model_class)
    self.assertEqual(saved_opts.source_vocab_path, loaded_opt.source_vocab_path)
    self.assertEqual(saved_opts.target_vocab_path, loaded_opt.target_vocab_path)


  def test_read_write_partial(self):
    saved_opts = training_utils.TrainOptions(
        hparams=self.hparams,
        model_class=None,
        source_vocab_path=self.source_vocab_file,
        target_vocab_path=None)
    saved_opts.dump(self.model_dir)

    loaded_opt = training_utils.TrainOptions.load(self.model_dir)

    self.assertEqual(saved_opts.hparams, loaded_opt.hparams)
    self.assertEqual(saved_opts.model_class, loaded_opt.model_class)
    self.assertEqual(saved_opts.source_vocab_path, loaded_opt.source_vocab_path)
    self.assertEqual(saved_opts.target_vocab_path, loaded_opt.target_vocab_path)


  def test_partial_read_write(self):
    pass


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

    self.assertEqual(
        set(features_.keys()),
        set(["source_tokens", "source_len"]))
    self.assertEqual(
        set(labels_.keys()),
        set(["target_tokens", "target_len"]))

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

  def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
    """Runs a multi-bleu test."""

    # Test with unicode
    result = training_utils.moses_multi_bleu(
        hypotheses=hypotheses,
        references=references,
        lowercase=lowercase)
    np.testing.assert_almost_equal(result, expected_bleu, decimal=2)

    # Test with byte string
    hypotheses_b = np.array([_.encode("utf-8") for _ in hypotheses]).astype("O")
    references_b = np.array([_.encode("utf-8") for _ in references]).astype("O")
    result = training_utils.moses_multi_bleu(
        hypotheses=hypotheses_b,
        references=references_b,
        lowercase=lowercase)
    np.testing.assert_almost_equal(result, expected_bleu, decimal=2)

  def test_multi_bleu(self):
    self._test_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over the dog 笑",
            "The brown fox jumps over the dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"]),
        lowercase=False,
        expected_bleu=46.51)

  def test_multi_bleu_lowercase(self):
    self._test_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over The Dog 笑",
            "The brown fox jumps over The Dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"]),
        lowercase=True,
        expected_bleu=46.51)

  def test_multi_bleu_with_eos(self):
    self._test_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over the dog 笑 SEQUENCE_END 2 x x x",
            "The brown fox jumps over the dog 2 笑 SEQUENCE_END 2 x x x"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"]),
        lowercase=False,
        expected_bleu=46.51)


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
