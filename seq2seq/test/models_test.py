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

from seq2seq import losses as seq2seq_losses
from seq2seq.data import vocab, input_pipeline
from seq2seq.training import utils as training_utils
from seq2seq.test import utils as test_utils
from seq2seq.models import BasicSeq2Seq, AttentionSeq2Seq
from seq2seq.contrib.seq2seq import helper as decode_helper

import tensorflow as tf
import numpy as np


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

  def tearDown(self):
    self.vocab_file.close()

  def create_model(self, mode, _params=None):
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

  def test_forward_pass(self):
    """Tests model forward pass by checking the shape of the outputs."""
    ex = self._create_example()
    helper = decode_helper.TrainingHelper(
        inputs=tf.convert_to_tensor(ex.target, dtype=tf.float32),
        sequence_length=tf.convert_to_tensor(ex.target_len, dtype=tf.int32))

    model = self.create_model(tf.contrib.learn.ModeKeys.TRAIN)
    decoder_output, _, _ = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decode_helper=helper)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(ex.target_len, max_decode_length)

    # Assert shapes are correct
    np.testing.assert_array_equal(decoder_output_.logits.shape, [
        np.max(expected_decode_len), self.batch_size,
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(
        decoder_output_.predicted_ids.shape,
        [np.max(expected_decode_len), self.batch_size])

  def test_inference(self):
    """Tests model inference by feeding dynamic inputs based on an embedding
    """
    model = self.create_model(tf.contrib.learn.ModeKeys.INFER)
    ex = self._create_example()

    embeddings = tf.get_variable(
        "W_embed", [model.target_vocab_info.total_size, self.input_depth])

    helper = decode_helper.GreedyEmbeddingHelper(
        embedding=embeddings,
        start_tokens=[0] * self.batch_size,
        end_token=-1)

    decoder_output, _, _ = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decode_helper=helper)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    # Assert shapes are correct
    expected_decode_len = model.params["inference.max_decode_length"]
    np.testing.assert_array_equal(decoder_output_.logits.shape, [
        expected_decode_len, self.batch_size,
        model.target_vocab_info.total_size
    ])
    np.testing.assert_array_equal(decoder_output_.predicted_ids.shape,
                                  [expected_decode_len, self.batch_size])

  def test_inference_with_beam_search(self):
    """Tests model inference by feeding dynamic inputs based on an embedding
      and using beam search to decode
    """
    self.batch_size = 1
    beam_width = 10

    ex = self._create_example()

    model = self.create_model(
        tf.contrib.learn.ModeKeys.INFER,
        {"inference.beam_search.beam_width": beam_width})

    embeddings = tf.get_variable(
        "W_embed", [model.target_vocab_info.total_size, self.input_depth])

    helper = decode_helper.GreedyEmbeddingHelper(
        embedding=embeddings,
        start_tokens=[0] * beam_width,
        end_token=-1)

    decoder_output, _, _ = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decode_helper=helper)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoder_output_ = sess.run(decoder_output)

    # Assert shapes are correct
    expected_decode_len = model.params["inference.max_decode_length"]
    np.testing.assert_array_equal(
        decoder_output_.predicted_ids.shape,
        [expected_decode_len, 1, beam_width])
    np.testing.assert_array_equal(
        decoder_output_.beam_search_output.beam_parent_ids.shape,
        [expected_decode_len, 1, beam_width])
    np.testing.assert_array_equal(
        decoder_output_.beam_search_output.scores.shape,
        [expected_decode_len, 1, beam_width])
    np.testing.assert_array_equal(
        decoder_output_.beam_search_output.original_outputs.predicted_ids.shape,
        [expected_decode_len, 1, beam_width])
    np.testing.assert_array_equal(
        decoder_output_.beam_search_output.original_outputs.logits.shape,
        [expected_decode_len, 1, beam_width,
         model.target_vocab_info.total_size])

  def test_gradients(self):
    """Ensures the parameter gradients can be computed and are not NaN
    """
    ex = self._create_example()

    helper = decode_helper.TrainingHelper(
        inputs=tf.convert_to_tensor(ex.target, dtype=tf.float32),
        sequence_length=tf.convert_to_tensor(ex.target_len, dtype=tf.int32))

    model = self.create_model(tf.contrib.learn.ModeKeys.TRAIN)
    decoder_output, _, _ = model.encode_decode(
        source=tf.convert_to_tensor(
            ex.source, dtype=tf.float32),
        source_len=tf.convert_to_tensor(
            ex.source_len, dtype=tf.int32),
        decode_helper=helper)

    # Get a loss to optimize
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits,
        targets=tf.ones_like(decoder_output.predicted_ids),
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
        [sources_file.name], [targets_file.name])
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

  def test_pipeline_train(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.TRAIN)
    predictions_, loss_, _ = fetches_

    target_len = self.sequence_length + 10 + 2
    max_decode_length = model.params["target.max_seq_len"]
    expected_decode_len = np.minimum(target_len, max_decode_length)

    np.testing.assert_array_equal(
        predictions_["logits"].shape,
        [self.batch_size, expected_decode_len - 1,
         model.target_vocab_info.total_size])
    np.testing.assert_array_equal(
        predictions_["losses"].shape,
        [self.batch_size, expected_decode_len - 1])
    np.testing.assert_array_equal(
        predictions_["predicted_ids"].shape,
        [self.batch_size, expected_decode_len - 1])
    self.assertFalse(np.isnan(loss_))


  def test_pipeline_inference(self):
    model, fetches_ = self._test_pipeline(tf.contrib.learn.ModeKeys.INFER)
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    np.testing.assert_array_equal(
        predictions_["logits"].shape,
        [self.batch_size, pred_len,
         model.target_vocab_info.total_size])
    np.testing.assert_array_equal(
        predictions_["predicted_ids"].shape,
        [self.batch_size, pred_len])

  def test_pipeline_beam_search_infer(self):
    self.batch_size = 1
    beam_width = 10
    model, fetches_ = self._test_pipeline(
        mode=tf.contrib.learn.ModeKeys.INFER,
        params={"inference.beam_search.beam_width": 10})
    predictions_, = fetches_
    pred_len = predictions_["predicted_ids"].shape[1]

    vocab_size = model.target_vocab_info.total_size
    np.testing.assert_array_equal(
        predictions_["predicted_ids"].shape,
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
    params_.update({
        "bridge.class": "PassThroughBridge",
        "encoder.params": {
            "rnn_cell": {
                "dropout_input_keep_prob": 0.8,
                "num_layers": 2,
                "residual_connections": True,
                "cell_class": "LSTMCell",
                "cell_params":  {"num_units": 12},
            }
        },
        "decoder.params": {
            "rnn_cell": {
                "num_layers": 2,
                "cell_class": "LSTMCell",
                "cell_params":  {"num_units": 12}
            }
        }
    })
    params_.update(params or {})
    return BasicSeq2Seq(
        source_vocab_info=self.vocab_info,
        target_vocab_info=self.vocab_info,
        params=params_,
        mode=mode)


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
    params_.update({
        "source.reverse": True
    })
    params_.update(params or {})
    return AttentionSeq2Seq(
        source_vocab_info=self.vocab_info,
        target_vocab_info=self.vocab_info,
        params=params_,
        mode=mode)


if __name__ == "__main__":
  tf.test.main()
