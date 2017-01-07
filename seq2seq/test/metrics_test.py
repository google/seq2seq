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
