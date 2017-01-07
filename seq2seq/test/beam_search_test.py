"""
Tests for Beam Search and related functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from seq2seq.inference import beam_search


class TestBeamStep(tf.test.TestCase):
  """Tests a single step of beam search
  """

  def setUp(self):
    super(TestBeamStep, self).setUp()
    self.state_size = 10
    config = beam_search.BeamSearchConfig(
        beam_width=3,
        vocab_size=5,
        eos_token=0,
        score_fn=beam_search.logprob_score,
        choose_successors_fn=beam_search.choose_top_k)
    self.config = config

  def test_step(self):
    beam_state = beam_search.BeamState(
        time=tf.constant(2),
        log_probs=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        scores=tf.zeros(self.config.beam_width),
        predicted_ids=tf.convert_to_tensor(
            [[1, 2, -1, -1, -1], [3, 4, -1, -1, -1], [5, 6, -1, -1, -1]]),
        beam_parent_ids=tf.zeros(self.config.beam_width))
    logits = tf.sparse_to_dense(
        [[0, 2], [0, 3], [1, 3], [1, 4]],
        output_shape=[self.config.beam_width, self.config.vocab_size],
        sparse_values=[1.9, 2.1, 3.1, 0.9],
        default_value=0.0001)
    next_beam_state = beam_search.beam_search_step(
        logits=logits, beam_state=beam_state, config=self.config)

    with self.test_session() as sess:
      res = sess.run(next_beam_state)

    expected_predictions = np.array(
        [[3, 4, 3, -1, -1], [1, 2, 3, -1, -1], [1, 2, 2, -1, -1]])
    np.testing.assert_array_equal(res.predicted_ids, expected_predictions)
    np.testing.assert_array_equal(res.beam_parent_ids, [1, 0, 0])

  def test_step_with_eos(self):
    beam_state = beam_search.BeamState(
        time=tf.constant(2),
        log_probs=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        scores=tf.nn.log_softmax(tf.ones(self.config.beam_width)),
        predicted_ids=tf.convert_to_tensor(
            [[1, 2, -1, -1, -1], [3, 0, -1, -1, -1], [5, 6, -1, -1, -1]]),
        beam_parent_ids=tf.zeros(self.config.beam_width))
    logits = tf.sparse_to_dense(
        [[0, 2], [1, 2], [2, 2]],
        output_shape=[self.config.beam_width, self.config.vocab_size],
        sparse_values=[1.0, 1.0, 1.0],
        default_value=0.0001)
    next_beam_state = beam_search.beam_search_step(
        logits=logits, beam_state=beam_state, config=self.config)

    with self.test_session() as sess:
      res = sess.run(next_beam_state)
      expected_predictions = np.array(
          [[3, 0, 0, -1, -1], [1, 2, 2, -1, -1], [5, 6, 2, -1, -1]])
      np.testing.assert_array_equal(res.predicted_ids, expected_predictions)
      previous_log_probs = sess.run(beam_state.log_probs)
      np.testing.assert_array_equal(res.log_probs[0], previous_log_probs[0])


class TestEosMasking(tf.test.TestCase):
  """Tests EOS masking used in beam search
  """

  def test_eos_masking(self):
    probs = tf.constant(
        [[-.2, -.2, -.2, -.2, -.2], [-.3, -.3, -.3, 3, 0], [5, 6, 0, 0, 0]])
    eos_token = 0
    previously_finished = tf.constant([0, 1, 0], dtype=tf.float32)
    masked = beam_search.mask_probs(probs, eos_token, previously_finished)

    with self.test_session() as sess:
      probs = sess.run(probs)
      masked = sess.run(masked)

      np.testing.assert_array_equal(probs[0], masked[0])
      np.testing.assert_array_equal(probs[2], masked[2])
      np.testing.assert_equal(masked[1][0], 0)
      np.testing.assert_approx_equal(masked[1][1], np.finfo('float32').min)
      np.testing.assert_approx_equal(masked[1][2], np.finfo('float32').min)
      np.testing.assert_approx_equal(masked[1][3], np.finfo('float32').min)
      np.testing.assert_approx_equal(masked[1][4], np.finfo('float32').min)


if __name__ == "__main__":
  tf.test.main()
