# -*- coding: utf-8 -*-

"""Collection of MetricSpecs for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import metric_spec
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops

from seq2seq.metrics import bleu

def accumulate_strings(values, name="strings"):
  """Accumulates strings into a vector.

  Args:
    values: A 1-d string tensor that contains values to add to the accumulator.

  Returns:
    A tuple (value_tensor, update_op).
  """
  tf.assert_type(values, tf.string)
  strings = tf.Variable(
      name=name,
      initial_value=[],
      dtype=tf.string,
      trainable=False,
      collections=[],
      validate_shape=True)
  value_tensor = tf.identity(strings)
  update_op = tf.assign(
      ref=strings,
      value=tf.concat([strings, values], 0),
      validate_shape=False)
  return value_tensor, update_op


class TextMetricSpec(metric_spec.MetricSpec):
  def __init__(self, name, separator=" ", eos_token="SEQUENCE_END"):
    """Initializer"""
    self.name = name
    self.separator = separator
    self.eos_token = eos_token

  def create_metric_ops(self, _inputs, labels, predictions):
    with variable_scope.variable_scope(self.name):
      # Join tokens into single strings
      predictions_flat = tf.reduce_join(
          predictions["predicted_tokens"], 1, separator=self.separator)
      labels_flat = tf.reduce_join(
          labels["target_tokens"], 1, separator=self.separator)

      sources_value, sources_update = accumulate_strings(
          values=predictions_flat, name="sources")
      targets_value, targets_update = accumulate_strings(
          values=labels_flat, name="targets")

      metric_value = tf.py_func(
          func=self.py_func,
          inp=[sources_value, targets_value],
          Tout=tf.float32,
          name="value")

    with tf.control_dependencies([sources_update, targets_update]):
      update_op = tf.identity(metric_value, name="update_op")

    return metric_value, update_op

  def py_func(self, hypotheses, references):
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
      hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
      references = np.char.encode(references, "utf-8")

    # Slice all hypotheses and references up to EOS
    sliced_hypotheses = [x.split(self.eos_token.encode("utf-8"))[0].strip()
                         for x in hypotheses]
    sliced_references = [x.split(self.eos_token.encode("utf-8"))[0].strip()
                         for x in references]

    # Strip special "@@ " tokens used for BPE
    # See https://github.com/rsennrich/subword-nmt
    # We hope this is rare enough that it will not have any adverse effects
    # on predicitons that do not use BPE
    sliced_hypotheses = [_.replace(b"@@ ", b"") for _ in sliced_hypotheses]
    sliced_references = [_.replace(b"@@ ", b"") for _ in sliced_references]

    # Convert back to unicode object
    sliced_hypotheses = [_.decode("utf-8") for _ in  sliced_hypotheses]
    sliced_references = [_.decode("utf-8") for _ in  sliced_references]

    return self.metric_fn(sliced_hypotheses, sliced_references)

  def metric_fn(self, hypotheses, references):
    raise NotImplementedError()


class BleuMetricSpec(TextMetricSpec):
  def __init__(self, separator=" ", eos_token="SEQUENCE_END"):
    super(BleuMetricSpec, self).__init__("bleu_metric", separator, eos_token)

  def metric_fn(self, hypotheses, references):
    return bleu.moses_multi_bleu(
        hypotheses,
        references,
        lowercase=False)


class LogPerplexityMetricSpec(metric_spec.MetricSpec):
  """A MetricSpec to calculate straming log perplexity"""
  def __init__(self):
    """Initializer"""
    pass

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates the metric op"""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(labels["target_len"] - 1),
        maxlen=tf.to_int32(tf.shape(predictions["losses"])[1]))
    return metrics.streaming_mean(predictions["losses"], loss_mask)


def streaming_log_perplexity():
  """Creates a MetricSpec that calculates the log perplexity.
  """
  return LogPerplexityMetricSpec()