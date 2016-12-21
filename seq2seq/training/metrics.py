"""Collection of metrics for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import metric_spec
from tensorflow.contrib.learn.python.learn import metric_spec

import numpy as np
from nltk.translate import bleu

def streaming_log_perplexity():
  """Creates a MetricSpec that calculates the log perplexity.
  """

  def perplexity_metric(losses, target_len):
    """Calculates the mean log perplexity based on losses and target_len"""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(target_len - 1),
        maxlen=tf.to_int32(tf.shape(losses)[1]))
    return metrics.streaming_mean(losses, loss_mask)

  return metric_spec.MetricSpec(
      metric_fn=perplexity_metric,
      label_key="target_len",
      prediction_key="losses")


def streaming_bleu():
  """Creates a MetricSpec that calculates the BLEU score.
  """

  def calculate_bleu(labels, predictions):
    """Calculates the BLEU score using NLTK."""
    score = bleu(labels.astype(str), predictions.astype(str))
    return np.array(score, dtype=np.float32)

  def bleu_metric(labels, predictions):
    """Calculates the BLEU score based on labels and predictions"""
    return tf.py_func(calculate_bleu, [labels, predictions],
                      [tf.float32])

  return metric_spec.MetricSpec(
      metric_fn=bleu_metric,
      label_key="labels",
      prediction_key="predictions")
