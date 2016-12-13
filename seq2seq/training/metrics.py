"""Collection of metrics for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import metric_spec

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
