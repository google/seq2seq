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
"""Tests for Models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import yaml
import numpy as np
import tensorflow as tf

from seq2seq.data import vocab, input_pipeline
from seq2seq.training import utils as training_utils
from seq2seq.test import utils as test_utils
from seq2seq.models import BasicSeq2Seq, AttentionSeq2Seq

TEST_PARAMS = yaml.load("""
embedding.dim: 5
encoder.params:
  rnn_cell:
    dropout_input_keep_prob: 0.8
    num_layers: 2
    residual_connections: True,
    cell_class: LSTMCell
    cell_params:
      num_units: 4
decoder.params:
  rnn_cell:
    num_layers: 2
    cell_class: LSTMCell
    cell_params:
      num_units: 4
""")


class EncoderDecoderTests(tf.test.TestCase):
  """Base class for EncoderDecoder tests. Tests for specific classes should
  inherit from this and tf.test.TestCase.
  """

  def setUp(self):
    super(EncoderDecoderTests, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 2
    self.input_depth = 4
    self.sequence_length = 10

    # Create vocabulary
    self.vocab_list = [str(_) for _ in range(10)]
    self.vocab_list += ["笑う", "泣く", "了解", "はい", "＾＿＾"]
    self.vocab_size = len(self.vocab_list)
    self.vocab_file = test_utils.create_temporary_vocab_file(self.vocab_list)
    self.vocab_info = vocab.get_vocab_info(self.vocab_file.name)

    tf.contrib.framework.get_or_create_global_step()

  def tearDown(self):
    self.vocab_file.close()

  def create_model(self, _mode, _params=None):
    """Creates model class to be tested. Subclasses must implement this method.
    """
    self.skipTest("Base module should not be tested.")

  def _create_example(self):
    """Creates example data for a test"""
    source = np.random.randn(self.batch_size, self.sequence_length,
                             self.input_depth)
    source_len = np.random.randint(0, self.sequence_length, [self.batch_size])
    target_len = np.random.randint(0, self.sequence_length * 2,
                                   [self.batch_size])
    target = np.random.randn(self.batch_size,
                             np.max(target_len), self.input_depth)
    labels = np.random.randint(0, self.vocab_size,
                               [self.batch_size, np.max(target_len) - 1])

    example_ = namedtuple(
        "Example", ["source", "source_len", "target", "target_len", "labels"])
    return example_(source, source_len, target, target_len, labels)

  def _test_pipeline(self, mode, params=None):
    """Helper function to test the full model pipeline.
    """
    # Create source and target example
    source_len = self.sequence_length + 5
    target_len = self.sequence_length + 10
    source = " ".join(np.random.choice(self.vocab_list, source_len))
    target = " ".join(np.random.choice(self.vocab_list, target_len))
    sources_file, targets_file = test_utils.create_temp_parallel_data(
        sources=[source], targets=[target])

    # Build model graph
    model = self.create_model(mode, params)
    input_pipeline_ = input_pipeline.ParallelTextInputPipeline(
        params={
            "source_files": [sources_file.name],
            "target_files": [targets_file.name]
        },
        mode=mode)
    input_fn = training_utils.create_input_fn(
        pipeline=input_pipeline_, batch_size=self.batch_size)
    features, labels = input_fn()
    fetches = model(features, labels, None)
    fetches = [_ for _ in fetches if _ is not None]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        fetches_ = sess.run(fetches)

    sources_file.close()
    targets_file.close()

    return model, fetches_

  def test_train(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.TRAIN)
    predictions_, loss_, _ = fetches_

    target_len = self.sequence_length + 10 + 2
    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(target_len, max_decode_length)

    np.testing.assert_array_equal(predictions_["logits"].shape, [
        self.batch_size, expected_decode_len - 1,
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(predictions_["losses"].shape,
                                  [self.batch_size, expected_decode_len - 1])
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [self.batch_size, expected_decode_len - 1])
    self.assertFalse(np.isnan(loss_))

  def test_infer(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.INFER)
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    np.testing.assert_array_equal(predictions_["logits"].shape, [
        self.batch_size, pred_len, model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [self.batch_size, pred_len])

  def test_infer_beam_search(self):
    self.batch_size = 1
    beam_width = 10
    model, fetches_ = self._test_pipeline(
        mode=tf.contrib.learn.ModeKeys.INFER,
        params={"inference.beam_search.beam_width": 10})
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    vocab_size = model.target_vocab_info.total_size
    np.testing.assert_array_equal(predictions_["predicted_ids"].shape,
                                  [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.beam_parent_ids"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.scores"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.original_outputs.predicted_ids"].shape,
        [1, pred_len, beam_width])
    np.testing.assert_array_equal(
        predictions_["beam_search_output.original_outputs.logits"].shape,
        [1, pred_len, beam_width, vocab_size])


class TestBasicSeq2Seq(EncoderDecoderTests):
  """Tests the seq2seq.models.BasicSeq2Seq model.
  """

  def setUp(self):
    super(TestBasicSeq2Seq, self).setUp()

  def create_model(self, mode, params=None):
    params_ = BasicSeq2Seq.default_params().copy()
    params_.update(TEST_PARAMS)
    params_.update({
        "vocab_source": self.vocab_file.name,
        "vocab_target": self.vocab_file.name,
        "bridge.class": "PassThroughBridge"
    })
    params_.update(params or {})
    return BasicSeq2Seq(params=params_, mode=mode)


class TestAttentionSeq2Seq(EncoderDecoderTests):
  """Tests the seq2seq.models.AttentionSeq2Seq model.
  """

  def setUp(self):
    super(TestAttentionSeq2Seq, self).setUp()
    self.encoder_rnn_cell = tf.contrib.rnn.LSTMCell(32)
    self.decoder_rnn_cell = tf.contrib.rnn.LSTMCell(32)
    self.attention_dim = 128

  def create_model(self, mode, params=None):
    params_ = AttentionSeq2Seq.default_params().copy()
    params_.update(TEST_PARAMS)
    params_.update({
        "source.reverse": True,
        "vocab_source": self.vocab_file.name,
        "vocab_target": self.vocab_file.name,
    })
    params_.update(params or {})
    return AttentionSeq2Seq(params=params_, mode=mode)


if __name__ == "__main__":
  tf.test.main()
