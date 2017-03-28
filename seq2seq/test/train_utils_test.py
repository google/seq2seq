# -*- coding: utf-8 -*-
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
Test Cases for Training utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.contrib import rnn_cell
from seq2seq.data import input_pipeline
from seq2seq.test import utils as test_utils
from seq2seq.training import utils as training_utils


class TestGetRNNCell(tf.test.TestCase):
  """Tests the get_rnn_cell function.
  """

  def test_single_layer(self):
    cell = training_utils.get_rnn_cell(
        cell_class="BasicLSTMCell", cell_params={"num_units": 16}, num_layers=1)
    self.assertIsInstance(cell, tf.contrib.rnn.BasicLSTMCell)
    self.assertEqual(cell.output_size, 16)

  def test_multi_layer(self):
    cell = training_utils.get_rnn_cell(
        cell_class="BasicLSTMCell", cell_params={"num_units": 16}, num_layers=2)
    self.assertIsInstance(cell, rnn_cell.ExtendedMultiRNNCell)
    self.assertEqual(cell.output_size, 16)

  def test_full_class_path(self):
    cell = training_utils.get_rnn_cell(
        cell_class="tensorflow.contrib.rnn.BasicRNNCell",
        cell_params={"num_units": 16},
        num_layers=1)
    self.assertIsInstance(cell, tf.contrib.rnn.BasicRNNCell)
    self.assertEqual(cell.output_size, 16)

  def test_dropout(self):
    cell = training_utils.get_rnn_cell(
        cell_class="BasicLSTMCell",
        cell_params={"num_units": 16},
        num_layers=1,
        dropout_input_keep_prob=0.5)
    self.assertIsInstance(cell, tf.contrib.rnn.DropoutWrapper)
    self.assertEqual(cell.output_size, 16)

  def test_extra_args(self):
    # Invalid args should raise a ValueError
    with self.assertRaises(ValueError):
      training_utils.get_rnn_cell(
          cell_class="LSTMCell",
          cell_params={"num_units": 16,
                       "use_peepholesERROR": True},
          num_layers=1)

    cell = training_utils.get_rnn_cell(
        cell_class="LSTMCell",
        cell_params={"num_units": 8,
                     "use_peepholes": True,
                     "forget_bias": 0.5},
        num_layers=1)
    self.assertIsInstance(cell, tf.contrib.rnn.LSTMCell)
    #pylint: disable=E1101,W0212
    self.assertEqual(cell._use_peepholes, True)
    self.assertEqual(cell._forget_bias, 0.5)
    self.assertEqual(cell.output_size, 8)


class TestTrainOptions(tf.test.TestCase):
  """Tests reading and writing of training options"""

  def setUp(self):
    super(TestTrainOptions, self).setUp()
    self.model_dir = tempfile.mkdtemp()
    self.model_params = {"num_layers": 4}
    self.model_class = "AttentionSeq2Seq"

  def test_read_write(self):
    saved_opts = training_utils.TrainOptions(
        model_class=self.model_class, model_params=self.model_params)
    saved_opts.dump(self.model_dir)

    loaded_opt = training_utils.TrainOptions.load(model_dir=self.model_dir)

    self.assertEqual(saved_opts.model_params, loaded_opt.model_params)
    self.assertEqual(saved_opts.model_class, loaded_opt.model_class)


class TestInputFn(tf.test.TestCase):
  """Tests create_input_fn"""

  def _test_with_args(self, **kwargs):
    """Helper function to test create_input_fn with keyword arguments"""
    sources_file, targets_file = test_utils.create_temp_parallel_data(
        sources=["Hello World ."], targets=["Goodbye ."])

    pipeline = input_pipeline.ParallelTextInputPipeline(
        params={
            "source_files": [sources_file.name],
            "target_files": [targets_file.name]
        },
        mode=tf.contrib.learn.ModeKeys.TRAIN)
    input_fn = training_utils.create_input_fn(pipeline=pipeline, **kwargs)
    features, labels = input_fn()

    with self.test_session() as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        features_, labels_ = sess.run([features, labels])

    self.assertEqual(
        set(features_.keys()), set(["source_tokens", "source_len"]))
    self.assertEqual(set(labels_.keys()), set(["target_tokens", "target_len"]))

  def test_without_buckets(self):
    self._test_with_args(batch_size=10)

  def test_wit_buckets(self):
    self._test_with_args(batch_size=10, bucket_boundaries=[0, 5, 10])


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
        stop_decay_at=1000.0,
        min_learning_rate=0.01,
        staircase=False)

    initial_lr = 1.0
    with self.test_session() as sess:
      # Should not decay past min_learning_rate
      np.testing.assert_almost_equal(sess.run(decay_fn(initial_lr, 900)), 0.01)


if __name__ == '__main__':
  tf.test.main()
