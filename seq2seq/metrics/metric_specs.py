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
"""Collection of MetricSpecs for training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import MetricSpec

from seq2seq.metrics import rouge
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
      ref=strings, value=tf.concat([strings, values], 0), validate_shape=False)
  return value_tensor, update_op


@six.add_metaclass(abc.ABCMeta)
class TextMetricSpec(MetricSpec):
  """Abstract class for text-based metrics calculated based on
  hypotheses and references. Subclasses must implement `metric_fn`.

  Args:
    name: A name for the metric
    separator: A seperator used to join predicted tokens. Default to space.
    eos_token: A string token used to find the end of a sequence. Hypotheses
      and references will be slcied until this token is found.
  """

  def __init__(self, name, separator=" ", eos_token="SEQUENCE_END"):
    # We don't call the super constructor on purpose
    #pylint: disable=W0231
    """Initializer"""
    self.name = name
    self.separator = separator
    self.eos_token = eos_token

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates (value, update_op) tensors
    """
    with tf.variable_scope(self.name):
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
          func=self._py_func,
          inp=[sources_value, targets_value],
          Tout=tf.float32,
          name="value")

    with tf.control_dependencies([sources_update, targets_update]):
      update_op = tf.identity(metric_value, name="update_op")

    return metric_value, update_op

  def _py_func(self, hypotheses, references):
    """Wrapper function that converts tensors to unicode and slices
      them until the EOS token is found.
    """
    # Deal with byte chars
    if hypotheses.dtype.kind == np.dtype("U"):
      hypotheses = np.char.encode(hypotheses, "utf-8")
    if references.dtype.kind == np.dtype("U"):
      references = np.char.encode(references, "utf-8")

    # Slice all hypotheses and references up to EOS
    sliced_hypotheses = [
        x.split(self.eos_token.encode("utf-8"))[0].strip() for x in hypotheses
    ]
    sliced_references = [
        x.split(self.eos_token.encode("utf-8"))[0].strip() for x in references
    ]

    # Strip special "@@ " tokens used for BPE
    # See https://github.com/rsennrich/subword-nmt
    # We hope this is rare enough that it will not have any adverse effects
    # on predicitons that do not use BPE
    sliced_hypotheses = [_.replace(b"@@ ", b"") for _ in sliced_hypotheses]
    sliced_references = [_.replace(b"@@ ", b"") for _ in sliced_references]

    # Convert back to unicode object
    sliced_hypotheses = [_.decode("utf-8") for _ in sliced_hypotheses]
    sliced_references = [_.decode("utf-8") for _ in sliced_references]

    return self.metric_fn(sliced_hypotheses, sliced_references)

  def metric_fn(self, hypotheses, references):
    """Calculates the value of the metric.

    Args:
      hypotheses: A python list of strings, each corresponding to a
        single hypothesis/example.
      references: A python list of strings, each corresponds to a single
        reference. Must have the same number of elements of `hypotheses`.

    Returns:
      A float value.
    """
    raise NotImplementedError()


class BleuMetricSpec(TextMetricSpec):
  """Calculates BLEU score using the Moses multi-bleu.perl script.
  """

  def __init__(self, separator=" ", eos_token="SEQUENCE_END"):
    super(BleuMetricSpec, self).__init__("bleu_metric", separator, eos_token)

  def metric_fn(self, hypotheses, references):
    return bleu.moses_multi_bleu(hypotheses, references, lowercase=False)


class RougeMetricSpec(TextMetricSpec):
  """Calculates BLEU score using the Moses multi-bleu.perl script.
  """

  def __init__(self, metric_name, **kwargs):
    super(RougeMetricSpec, self).__init__(metric_name, **kwargs)
    self.metric_name = metric_name

  def metric_fn(self, hypotheses, references):
    if not hypotheses or not references:
      return np.float32(0.0)
    return np.float32(rouge.rouge(hypotheses, references)[self.metric_name])


class LogPerplexityMetricSpec(MetricSpec):
  """A MetricSpec to calculate straming log perplexity"""

  def __init__(self):
    """Initializer"""
    # We don't call the super constructor on purpose
    #pylint: disable=W0231
    pass

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates the metric op"""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(labels["target_len"] - 1),
        maxlen=tf.to_int32(tf.shape(predictions["losses"])[1]))
    return metrics.streaming_mean(predictions["losses"], loss_mask)


METRIC_SPECS_DICT = {
    "bleu": BleuMetricSpec(),
    "log_perplexity": LogPerplexityMetricSpec(),
    "rouge_1/f_score": RougeMetricSpec("rouge_1/f_score"),
    "rouge_1/r_score": RougeMetricSpec("rouge_1/r_score"),
    "rouge_1/p_score": RougeMetricSpec("rouge_1/p_score"),
    "rouge_2/f_score": RougeMetricSpec("rouge_2/f_score"),
    "rouge_2/r_score": RougeMetricSpec("rouge_2/r_score"),
    "rouge_2/p_score": RougeMetricSpec("rouge_2/p_score"),
    "rouge_l/f_score": RougeMetricSpec("rouge_l/f_score")
}
