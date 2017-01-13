# -*- coding: utf-8 -*-

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


class SuperLSTMCellTest(tf.test.TestCase):
  """Tests the SuperLSTMCell"""
  def test_cell(self):
    inputs = tf.constant(np.random.randn(1, 2), dtype=tf.float32)
    state = (
        tf.constant(np.random.randn(1, 2), dtype=tf.float32),
        tf.constant(np.random.randn(1, 2), dtype=tf.float32))
    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      cell = rnn_cell.SuperLSTMCell(2, use_layer_norm=True)
      res = cell(inputs, state)

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_ = sess.run(res)

    self.assertEqual(res_[0].shape, (1, 2))
    self.assertEqual(res_[1][0].shape, (1, 2))
    self.assertEqual(res_[1][1].shape, (1, 2))


class HyperLSTMCellTest(tf.test.TestCase):
  """Tests the HyperLSTMCellTest"""
  def test_cell(self):
    inputs = tf.constant(np.random.randn(1, 2), dtype=tf.float32)
    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      cell = rnn_cell.HyperLSTMCell(2, use_layer_norm=True, hyper_num_units=2)
      state = cell.zero_state(1, dtype=tf.float32)
      res = cell(inputs, state)

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_ = sess.run(res)

    self.assertEqual(res_[0].shape, (1, 2))
    self.assertEqual(res_[1][0].shape, (1, 4))
    self.assertEqual(res_[1][1].shape, (1, 4))


class ExtendedMultiRNNCellTest(tf.test.TestCase):
  """Tests the ExtendedMultiRNNCell"""

  def test_without_residuals(self):
    inputs = tf.constant(np.random.randn(1, 2))
    state = (
        tf.constant(np.random.randn(1, 2)),
        tf.constant(np.random.randn(1, 2)))

    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      standard_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.GRUCell(2)] * 2,
          state_is_tuple=True)
      res_standard = standard_cell(inputs, state, scope="standard")

      test_cell = rnn_cell.ExtendedMultiRNNCell(
          [tf.contrib.rnn.GRUCell(2)] * 2)
      res_test = test_cell(inputs, state, scope="test")

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_standard_, res_test_, = sess.run([res_standard, res_test])

    # Make sure it produces the same results as the standard cell
    self.assertAllClose(res_standard_[0], res_test_[0])
    self.assertAllClose(res_standard_[1][0], res_test_[1][0])
    self.assertAllClose(res_standard_[1][1], res_test_[1][1])

  def test_with_residuals(self):
    inputs = tf.constant(np.random.randn(1, 2))
    state = (
        tf.constant(np.random.randn(1, 2)),
        tf.constant(np.random.randn(1, 2)))

    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      test_cell = rnn_cell.ExtendedMultiRNNCell(
          [tf.contrib.rnn.GRUCell(2)] * 2,
          residual_connections=True)
      res_test = test_cell(inputs, state, scope="test")

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_test_ = sess.run(res_test)

    # Just a smoke test, these numbers are not calculated by hand
    self.assertAllClose([[-2.10551531, -1.03866035]], res_test_[0])
    self.assertAllClose([[-0.7694797, -0.1046757]], res_test_[1][0])
    self.assertAllClose([[-0.89064711, -0.61222793]], res_test_[1][1])

  def test_with_residuals_transform(self):
    inputs = tf.constant(np.random.randn(1, 5))
    state = (
        tf.constant(np.random.randn(1, 2)),
        tf.constant(np.random.randn(1, 2)))

    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      test_cell = rnn_cell.ExtendedMultiRNNCell(
          [tf.contrib.rnn.GRUCell(2)] * 2,
          residual_connections=True)
      res_test = test_cell(inputs, state, scope="test")

    with self.test_session() as sess:
      sess.run([tf.global_variables_initializer()])
      res_test_ = sess.run(res_test)

    # Just a smoke test, these numbers are not calculated by hand
    self.assertAllClose([[-0.18053266, -0.45155333]], res_test_[0])
    self.assertAllClose([[-0.3921121, -0.56264944]], res_test_[1][0])
    self.assertAllClose([[0.58282027, 0.29629828]], res_test_[1][1])


if __name__ == "__main__":
  tf.test.main()
