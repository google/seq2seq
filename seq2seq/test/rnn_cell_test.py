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
Unit tests for input-related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from seq2seq.contrib import rnn_cell

import numpy as np


class ExtendedMultiRNNCellTest(tf.test.TestCase):
  """Tests the ExtendedMultiRNNCell"""

  def test_without_residuals(self):
    inputs = tf.constant(np.random.randn(1, 2))
    state = (tf.constant(np.random.randn(1, 2)),
             tf.constant(np.random.randn(1, 2)))

    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      standard_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.GRUCell(2) for _ in range(2)], state_is_tuple=True)
      res_standard = standard_cell(inputs, state, scope="standard")

      test_cell = rnn_cell.ExtendedMultiRNNCell(
          [tf.contrib.rnn.GRUCell(2) for _ in range(2)])
      res_test = test_cell(inputs, state, scope="test")

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_standard_, res_test_, = sess.run([res_standard, res_test])

    # Make sure it produces the same results as the standard cell
    self.assertAllClose(res_standard_[0], res_test_[0])
    self.assertAllClose(res_standard_[1][0], res_test_[1][0])
    self.assertAllClose(res_standard_[1][1], res_test_[1][1])

  def _test_with_residuals(self, inputs, **kwargs):
    """Runs the cell in a session"""
    inputs = tf.convert_to_tensor(inputs)
    state = (tf.constant(np.random.randn(1, 2)),
             tf.constant(np.random.randn(1, 2)))

    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      test_cell = rnn_cell.ExtendedMultiRNNCell(
          [tf.contrib.rnn.GRUCell(2) for _ in range(2)],
          residual_connections=True,
          **kwargs)
      res_test = test_cell(inputs, state, scope="test")

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      return sess.run(res_test)

  def _test_constant_shape(self, combiner):
    """Tests a residual combiner whose shape doesn't change
    with depth"""
    inputs = np.random.randn(1, 2)
    with tf.variable_scope("same_input_size"):
      res_ = self._test_with_residuals(inputs, residual_combiner=combiner)
      self.assertEqual(res_[0].shape, (1, 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    inputs = np.random.randn(1, 5)
    with tf.variable_scope("diff_input_size"):
      res_ = self._test_with_residuals(inputs, residual_combiner=combiner)
      self.assertEqual(res_[0].shape, (1, 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    with tf.variable_scope("same_input_size_dense"):
      res_ = self._test_with_residuals(
          inputs, residual_combiner=combiner, residual_dense=True)
      self.assertEqual(res_[0].shape, (1, 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    inputs = np.random.randn(1, 5)
    with tf.variable_scope("diff_input_size_dense"):
      res_ = self._test_with_residuals(
          inputs, residual_combiner=combiner, residual_dense=True)
      self.assertEqual(res_[0].shape, (1, 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

  def test_residuals_mean(self):
    self._test_constant_shape(combiner="mean")

  def test_residuals_add(self):
    self._test_constant_shape(combiner="add")

  def test_residuals_concat(self):
    inputs = np.random.randn(1, 2)
    with tf.variable_scope("same_input_size"):
      res_ = self._test_with_residuals(inputs, residual_combiner="concat")
      self.assertEqual(res_[0].shape, (1, 6))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    inputs = np.random.randn(1, 5)
    with tf.variable_scope("diff_input_size"):
      res_ = self._test_with_residuals(inputs, residual_combiner="concat")
      self.assertEqual(res_[0].shape, (1, 5 + 2 + 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    inputs = np.random.randn(1, 2)
    with tf.variable_scope("same_input_size_dense"):
      res_ = self._test_with_residuals(
          inputs, residual_combiner="concat", residual_dense=True)
      self.assertEqual(res_[0].shape, (1, 2 + 4 + 2))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))

    inputs = np.random.randn(1, 5)
    with tf.variable_scope("diff_input_size_dense"):
      res_ = self._test_with_residuals(
          inputs, residual_combiner="concat", residual_dense=True)
      self.assertEqual(res_[0].shape, (1, 2 + (5 + 2) + 5))
      self.assertEqual(res_[1][0].shape, (1, 2))
      self.assertEqual(res_[1][1].shape, (1, 2))


if __name__ == "__main__":
  tf.test.main()
