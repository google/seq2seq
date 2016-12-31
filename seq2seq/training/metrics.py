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
from nltk.translate import bleu_score


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

  def calculate_bleu(target_ids, predictions):
    """Calculates the BLEU score using NLTK."""
    # NLTK expects the inputs as strings; we also use corpus_bleu
    # to calculate the BLEU score for all sentences in a batch
    # and wrap the target_ids in a list, as NLTK expects potentially
    # multiple reference translations per example
    str_target_ids = [[list(sentence_labels.astype(str))]
                      for sentence_labels in list(target_ids)]
    str_predictions = [list(sentence_predictions.astype(str))
                       for sentence_predictions in list(predictions)]
    score = bleu_score.corpus_bleu(str_target_ids, str_predictions)
    return np.array(score, dtype=np.float32)

  def bleu_metric(target_ids, predictions):
    """Calculates the BLEU score based on labels and predictions"""
    print('Calculating BLEU metric...')
    bleu_scores = tf.py_func(calculate_bleu, [target_ids, predictions],
                            [tf.float32])
    return metrics.streaming_mean(bleu_scores)

  return metric_spec.MetricSpec(
      metric_fn=bleu_metric,
      label_key="target_ids",
      prediction_key="predictions")
