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
from __future__ import unicode_literals

from pydoc import locate
import abc

import numpy as np
import six

import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import MetricSpec

from seq2seq.data import postproc
from seq2seq.configurable import Configurable
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
class TextMetricSpec(Configurable, MetricSpec):
  """Abstract class for text-based metrics calculated based on
  hypotheses and references. Subclasses must implement `metric_fn`.

  Args:
    name: A name for the metric
    separator: A separator used to join predicted tokens. Default to space.
    eos_token: A string token used to find the end of a sequence. Hypotheses
      and references will be slcied until this token is found.
  """

  def __init__(self, params, name):
    # We don't call the super constructor on purpose
    #pylint: disable=W0231
    """Initializer"""
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.EVAL)
    self._name = name
    self._eos_token = self.params["eos_token"]
    self._sos_token = self.params["sos_token"]
    self._separator = self.params["separator"]
    self._postproc_fn = None
    if self.params["postproc_fn"]:
      self._postproc_fn = locate(self.params["postproc_fn"])
      if self._postproc_fn is None:
        raise ValueError("postproc_fn not found: {}".format(
            self.params["postproc_fn"]))

  @property
  def name(self):
    """Name of the metric"""
    return self._name

  @staticmethod
  def default_params():
    return {
        "sos_token": "SEQUENCE_START",
        "eos_token": "SEQUENCE_END",
        "separator": " ",
        "postproc_fn": "",
    }

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates (value, update_op) tensors
    """
    with tf.variable_scope(self._name):

      # Join tokens into single strings
      predictions_flat = tf.reduce_join(
          predictions["predicted_tokens"], 1, separator=self._separator)
      labels_flat = tf.reduce_join(
          labels["target_tokens"], 1, separator=self._separator)

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

    # Convert back to unicode object
    hypotheses = [_.decode("utf-8") for _ in hypotheses]
    references = [_.decode("utf-8") for _ in references]

    # Slice all hypotheses and references up to SOS -> EOS
    sliced_hypotheses = [postproc.slice_text(
        _, self._eos_token, self._sos_token) for _ in hypotheses]
    sliced_references = [postproc.slice_text(
        _, self._eos_token, self._sos_token) for _ in references]

    # Apply postprocessing function
    if self._postproc_fn:
      sliced_hypotheses = [self._postproc_fn(_) for _ in sliced_hypotheses]
      sliced_references = [self._postproc_fn(_) for _ in sliced_references]

    return self.metric_fn(sliced_hypotheses, sliced_references) #pylint: disable=E1102

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

  def __init__(self, params):
    super(BleuMetricSpec, self).__init__(params, "bleu")

  def metric_fn(self, hypotheses, references):
    return bleu.moses_multi_bleu(hypotheses, references, lowercase=False)


class RougeMetricSpec(TextMetricSpec):
  """Calculates BLEU score using the Moses multi-bleu.perl script.
  """

  def __init__(self, params, **kwargs):
    if not params["rouge_type"]:
      raise ValueError("You must provide a rouge_type for ROUGE")
    super(RougeMetricSpec, self).__init__(
        params, params["rouge_type"], **kwargs)
    self._rouge_type = self.params["rouge_type"]

  @staticmethod
  def default_params():
    params = TextMetricSpec.default_params()
    params.update({
        "rouge_type": "",
    })
    return params

  def metric_fn(self, hypotheses, references):
    if not hypotheses or not references:
      return np.float32(0.0)
    return np.float32(rouge.rouge(hypotheses, references)[self._rouge_type])


class LogPerplexityMetricSpec(MetricSpec, Configurable):
  """A MetricSpec to calculate straming log perplexity"""

  def __init__(self, params):
    """Initializer"""
    # We don't call the super constructor on purpose
    #pylint: disable=W0231
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.EVAL)

  @staticmethod
  def default_params():
    return {}

  @property
  def name(self):
    """Name of the metric"""
    return "log_perplexity"

  def create_metric_ops(self, _inputs, labels, predictions):
    """Creates the metric op"""
    loss_mask = tf.sequence_mask(
        lengths=tf.to_int32(labels["target_len"] - 1),
        maxlen=tf.to_int32(tf.shape(predictions["losses"])[1]))
    return metrics.streaming_mean(predictions["losses"], loss_mask)
