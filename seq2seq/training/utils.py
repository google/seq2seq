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

# -*- coding: utf-8 -*-

"""Miscellaneous training utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import os
from collections import defaultdict
from pydoc import locate

import json

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.contrib import rnn_cell
from seq2seq.training import hooks

class TrainOptions(object):
  """A collectionf of options that are passed to the training script
  and should be saved to perform inference later on.

  Args:
    model_dir: The model directory. Options will be dumped in this
      directory.
    hparams: A dictionary of hyperparameter values.
    model_class: The model class name, a string.
    source_vocab_path: Path to the source vocabulary
    target_vocab_path: Path to the target vocabulary
  """
  def __init__(self,
               hparams=None,
               model_class=None,
               source_vocab_path=None,
               target_vocab_path=None):
    self.hparams = hparams
    self.model_class = model_class
    self.source_vocab_path = source_vocab_path
    self.target_vocab_path = target_vocab_path

  @staticmethod
  def path(model_dir):
    """Returns the path to the options file.

    Args:
      model_dir: The model directory
    """
    return os.path.join(model_dir, "train_options.json")

  def dump(self, model_dir):
    """Dumps the options to a file in the model directory.

    Args:
      model_dir: Path to the model directory. The options will be
      dumped into a file in this directory.
    """
    gfile.MakeDirs(model_dir)
    options_dict = {
        "hparams": self.hparams,
        "model_class": self.model_class,
        "source_vocab_path": self.source_vocab_path,
        "target_vocab_path": self.target_vocab_path
    }

    with gfile.GFile(TrainOptions.path(model_dir), "w") as file:
      file.write(json.dumps(options_dict).encode("utf-8"))

  @staticmethod
  def load(model_dir):
    """ Loads options from the given model directory.

    Args:
      model_dir: Path to the model directory.
    """
    with gfile.GFile(TrainOptions.path(model_dir), "r") as file:
      options_dict = json.loads(file.read().decode("utf-8"))
    options_dict = defaultdict(None, options_dict)

    return TrainOptions(
        hparams=options_dict["hparams"],
        model_class=options_dict["model_class"],
        source_vocab_path=options_dict["source_vocab_path"],
        target_vocab_path=options_dict["target_vocab_path"])

def cell_from_spec(cell_classname, cell_params):
  """Create a RNN Cell instance from a JSON string.

  Args:
    cell_classname: Name of the cell class, e.g. "BasicLSTMCell".
    cell_params: A dictionary of parameters to pass to the cell constructor.

  Returns:
    A RNNCell instance.
  """

  cell_params = cell_params.copy()

  # Find the cell class
  cell_class = locate(cell_classname) or getattr(rnn_cell, cell_classname)

  # Make sure additional arguments are valid
  cell_args = set(inspect.getargspec(cell_class.__init__).args[1:])
  for key in cell_params.keys():
    if key not in cell_args:
      raise ValueError(
          """{} is not a valid argument for {} class. Available arguments
          are: {}""".format(key, cell_class.__name__, cell_args))

  # Create cell
  return cell_class(**cell_params)


def get_rnn_cell(cell_class,
                 cell_params,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
  """Creates a new RNN Cell

  Args:
    cell_class: Name of the cell class, e.g. "BasicLSTMCell".
    cell_params: A dictionary of parameters to pass to the cell constructor.
    num_layers: Number of layers. The cell will be wrapped with
      `tf.contrib.rnn.MultiRNNCell`
    dropout_input_keep_prob: Dropout keep probability applied
      to the input of cell *at each layer*
    dropout_output_keep_prob: Dropout keep probability applied
      to the output of cell *at each layer*
    residual_connections: If true, add residual connections
      between all cells

  Returns:
    An instance of `tf.contrib.rnn.RNNCell`.
  """

  #pylint: disable=redefined-variable-type
  cells = []
  for _ in range(num_layers):
    cell = cell_from_spec(cell_class, cell_params)
    if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell=cell,
          input_keep_prob=dropout_input_keep_prob,
          output_keep_prob=dropout_output_keep_prob)
    cells.append(cell)

  if len(cells) > 1:
    final_cell = rnn_cell.ExtendedMultiRNNCell(
        cells=cells,
        residual_connections=residual_connections,
        residual_combiner=residual_combiner,
        residual_dense=residual_dense)
  else:
    final_cell = cells[0]

  return final_cell


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False):
  """Creates a function that decays the learning rate.

  Args:
    decay_steps: How often to apply decay.
    decay_rate: A Python number. The decay rate.
    start_decay_at: Don't decay before this step
    stop_decay_at: Don't decay after this step
    min_learning_rate: Don't decay below this number
    decay_type: A decay function name defined in `tf.train`
    staircase: Whether to apply decay in a discrete staircase,
      as opposed to continuous, fashion.

  Returns:
    A function that takes (learning_rate, global_step) as inputs
    and returns the learning rate for the given step.
    Returns `None` if decay_type is empty or None.
  """
  if decay_type is None or decay_type == "":
    return None

  def decay_fn(learning_rate, global_step):
    """The computed learning rate decay function.
    """
    decay_type_fn = getattr(tf.train, decay_type)
    decayed_learning_rate = decay_type_fn(
        learning_rate=learning_rate,
        global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
        name="decayed_learning_rate")

    final_lr = tf.train.piecewise_constant(
        x=global_step,
        boundaries=[start_decay_at],
        values=[learning_rate, decayed_learning_rate])

    if min_learning_rate:
      final_lr = tf.maximum(final_lr, min_learning_rate)

    return final_lr

  return decay_fn


def create_input_fn(pipeline,
                    batch_size,
                    bucket_boundaries=None,
                    allow_smaller_final_batch=False):
  """Creates an input function that can be used with tf.learn estimators.
    Note that you must pass "factory funcitons" for both the data provider and
    featurizer to ensure that everything will be created in  the same graph.

  Args:
    pipeline: An instance of `seq2seq.data.InputPipeline`.
    batch_size: Create batches of this size. A queue to hold a
      reasonable number of batches in memory is created.
    bucket_boundaries: int list, increasing non-negative numbers.
      If None, no bucket is performed.

  Returns:
    An input function that returns `(feature_batch, labels_batch)`
    tuples when called.
  """

  def input_fn():
    """Creates features and labels.
    """

    data_provider = pipeline.make_data_provider()
    features_and_labels = pipeline.read_from_data_provider(data_provider)

    if bucket_boundaries:
      _, batch = tf.contrib.training.bucket_by_sequence_length(
          input_length=features_and_labels["source_len"],
          bucket_boundaries=bucket_boundaries,
          tensors=features_and_labels,
          batch_size=batch_size,
          keep_input=features_and_labels["source_len"] >= 1,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          allow_smaller_final_batch=allow_smaller_final_batch,
          name="bucket_queue")
    else:
      batch = tf.train.batch(
          tensors=features_and_labels,
          enqueue_many=False,
          batch_size=batch_size,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          allow_smaller_final_batch=allow_smaller_final_batch,
          name="batch_queue")

    # Separate features and labels
    features_batch = {k: batch[k] for k in pipeline.feature_keys}
    if set(batch.keys()).intersection(pipeline.label_keys):
      labels_batch = {k: batch[k] for k in pipeline.label_keys}
    else:
      labels_batch = None

    return features_batch, labels_batch

  return input_fn


def create_default_training_hooks(
    estimator,
    sample_frequency=500,
    source_delimiter=" ",
    target_delimiter=" "):
  """Creates common SessionRunHooks used for training.

  Args:
    estimator: The estimator instance
    sample_frequency: frequency of samples passed to the TrainSampleHook

  Returns:
    An array of `SessionRunHook` items.
  """
  output_dir = estimator.model_dir
  training_hooks = []

  model_analysis_hook = hooks.PrintModelAnalysisHook(
      filename=os.path.join(output_dir, "model_analysis.txt"))
  training_hooks.append(model_analysis_hook)

  train_sample_hook = hooks.TrainSampleHook(
      every_n_steps=sample_frequency,
      sample_dir=os.path.join(output_dir, "samples"),
      source_delimiter=source_delimiter,
      target_delimiter=target_delimiter)
  training_hooks.append(train_sample_hook)

  metadata_hook = hooks.MetadataCaptureHook(
      output_dir=os.path.join(output_dir, "metadata"),
      step=10)
  training_hooks.append(metadata_hook)

  tokens_per_sec_counter = hooks.TokensPerSecondCounter(
      every_n_steps=100,
      output_dir=output_dir)
  training_hooks.append(tokens_per_sec_counter)

  return training_hooks
