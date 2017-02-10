# -*- coding: utf-8 -*-

"""Tests for Metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from seq2seq.metrics import bleu
from seq2seq.metrics.metric_specs import BleuMetricSpec

class TestMosesBleu(tf.test.TestCase):
  """Tests using the Moses multi-bleu script to calcualte BLEU score"""

  def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
    """Runs a multi-bleu test."""
    result = bleu.moses_multi_bleu(
        hypotheses=hypotheses,
        references=references,
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


class TestBleuMetricSpec(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""

  def test_bleu(self):
    predictions = {
        "predicted_tokens": tf.placeholder(dtype=tf.string)
    }
    labels = {
        "target_tokens": tf.placeholder(dtype=tf.string)
    }

    metric_spec = BleuMetricSpec()
    value, update_op = metric_spec.create_metric_ops(None, labels, predictions)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      hypotheses = ["A B C D E F", "A B C D E F"]
      references = ["A B C D E F", "A B A D E F"]

      scores = []
      for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split(" ")
        ref = ref.split(" ")
        sess.run(update_op, {
            predictions["predicted_tokens"]: [hyp],
            labels["target_tokens"]: [ref]
        })
        scores.append(sess.run(value))

      np.testing.assert_almost_equal(scores[0], 100.0, decimal=2)
      np.testing.assert_almost_equal(scores[1], 69.19, decimal=2)


if __name__ == "__main__":
  tf.test.main()
