# -*- coding: utf-8 -*-

"""Tests for Metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from seq2seq.training import metrics
from seq2seq.metrics import bleu

class TestMosesBleu(tf.test.TestCase):
  """Tests using the Moses multi-bleu script to calcualte BLEU score"""

  def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
    """Runs a multi-bleu test."""

    # Test with unicode
    result = bleu.moses_multi_bleu(
        hypotheses=hypotheses,
        references=references,
        lowercase=lowercase)
    np.testing.assert_almost_equal(result, expected_bleu, decimal=2)

    # Test with byte string
    hypotheses_b = np.array([_.encode("utf-8") for _ in hypotheses]).astype("O")
    references_b = np.array([_.encode("utf-8") for _ in references]).astype("O")
    result = bleu.moses_multi_bleu(
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


class TestBleuMetric(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""

  def test_bleu(self):
    predictions = tf.placeholder(dtype=tf.string)
    targets = tf.placeholder(dtype=tf.string)

    value, update_op = metrics.streaming_bleu(predictions, targets)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      hypotheses = ["A B C D E F", "A B C D E F"]
      references = ["A B C D E F", "A B A D E F"]

      scores = []
      for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split(" ")
        ref = ref.split(" ")
        sess.run(update_op, {predictions: [hyp], targets: [ref]})
        scores.append(sess.run(value))

      np.testing.assert_almost_equal(scores[0], 100.0, decimal=2)
      np.testing.assert_almost_equal(scores[1], 69.19, decimal=2)


if __name__ == "__main__":
  tf.test.main()
