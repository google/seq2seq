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
"""Base class for models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import tensorflow as tf

from seq2seq.configurable import Configurable
from seq2seq.training import utils as training_utils


def _flatten_dict(dict_, parent_key="", sep="."):
  """Flattens a nested dictionary. Namedtuples within
  the dictionary are converted to dicts.

  Args:
    dict_: The dictionary to flatten.
    parent_key: A prefix to prepend to each key.
    sep: Separator between parent and child keys, a string. For example
      { "a": { "b": 3 } } will become { "a.b": 3 } if the separator is ".".

  Returns:
    A new flattened dictionary.
  """
  items = []
  for key, value in dict_.items():
    new_key = parent_key + sep + key if parent_key else key
    if isinstance(value, collections.MutableMapping):
      items.extend(_flatten_dict(value, new_key, sep=sep).items())
    elif isinstance(value, tuple) and hasattr(value, "_asdict"):
      dict_items = collections.OrderedDict(zip(value._fields, value))
      items.extend(_flatten_dict(dict_items, new_key, sep=sep).items())
    else:
      items.append((new_key, value))
  return dict(items)


class ModelBase(Configurable):
  """Abstract base class for models.

  Args:
    params: A dictionary of hyperparameter values
    name: A name for this model to be used as a variable scope
  """

  def __init__(self, params, mode, name):
    self.name = name
    Configurable.__init__(self, params, mode)

  def _clip_gradients(self, grads_and_vars):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, self.params["optimizer.clip_gradients"])
    return list(zip(clipped_gradients, variables))

  def _build_train_op(self, loss):
    """Creates the training operation"""
    learning_rate_decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type=self.params["optimizer.lr_decay_type"] or None,
        decay_steps=self.params["optimizer.lr_decay_steps"],
        decay_rate=self.params["optimizer.lr_decay_rate"],
        start_decay_at=self.params["optimizer.lr_start_decay_at"],
        stop_decay_at=self.params["optimizer.lr_stop_decay_at"],
        min_learning_rate=self.params["optimizer.lr_min_learning_rate"],
        staircase=self.params["optimizer.lr_staircase"])

    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=self.params["optimizer.learning_rate"],
        learning_rate_decay_fn=learning_rate_decay_fn,
        clip_gradients=self._clip_gradients,
        optimizer=self.params["optimizer.name"],
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

  @staticmethod
  def default_params():
    """Returns a dictionary of default parameters for this model."""
    return {
        "optimizer.name": "Adam",
        "optimizer.learning_rate": 1e-4,
        "optimizer.lr_decay_type": "",
        "optimizer.lr_decay_steps": 100,
        "optimizer.lr_decay_rate": 0.99,
        "optimizer.lr_start_decay_at": 0,
        "optimizer.lr_stop_decay_at": tf.int32.max,
        "optimizer.lr_min_learning_rate": 1e-12,
        "optimizer.lr_staircase": False,
        "optimizer.clip_gradients": 5.0,
        "training.data_parallelism": 1
    }

  def batch_size(self, features, labels):
    """Returns the batch size for a batch of examples"""
    raise NotImplementedError()

  def _build_parallel(self, features, labels, params):
    """Builds one or more model replicas on GPU devices. If
    `training.data_parallelism` is set to 1 this function does nothing and
    just calls the build method.

    If `training.data_parallelism` is > 1 and not enough GPUs are
    available this will throw an error.

    If `training.data_parallelism` = N > 1 and enough GPUs are available this
    will create a model replica on each GPU andsplit the training batch into
    N pieces, merge the predictions and average the losses.

    If model is not in training mode this does nothing and just calls the
    build method.
    """
    parallelism = self.params["training.data_parallelism"]

    # Data parallelism is disabled
    if parallelism <= 0:
      return self._build(features, labels, params)

    # Not training
    if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
      return self._build(features, labels, params)

    # Data parallelism is enabled
    available_gpus = training_utils.get_available_gpus()
    tf.logging.info("Available GPUs: %s", available_gpus)

    # Make sure we have enough GPUs
    if len(available_gpus) < parallelism:
      raise ValueError(
          "Data Parallelism set to {}, but only {} GPUs available""".format(
              parallelism, len(available_gpus)))

    # Split all features and labels
    features_split = {k: tf.split(v, parallelism) for k, v in features.items()}
    labels_split = {k: tf.split(v, parallelism) for k, v in labels.items()}
    tf.logging.info(features_split)

    scope = tf.get_variable_scope()

    all_losses = []
    all_predictions = []
    for idx in range(parallelism):
      # Share variables on all replicas
      if idx > 0:
        scope.reuse_variables()
      # Create each model replica
      gpu_device = available_gpus[idx]
      tf.logging.info("Creating replica %d on device %s", idx, gpu_device)
      with tf.device(gpu_device):
        tf.logging.info(idx)
        rep_features = {k: v[idx] for k, v in features_split.items()}
        rep_labels = {k: v[idx] for k, v in labels_split.items()}
        rep_pred, rep_loss = self._build(rep_features, rep_labels, params)
        all_losses.append(rep_loss)
        all_predictions.append(rep_pred)

    # Concat all predictions
    prediction_keys = all_predictions[0].keys()
    predictions = {
        k: tf.concat([_[k] for _ in all_predictions], 0)
        for k in prediction_keys
    }

    # Take the average loss
    loss = tf.reduce_mean(all_losses)

    return predictions, loss


  def __call__(self, features, labels, params):
    """Creates the model graph. See the model_fn documentation in
    tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    with tf.variable_scope("model"):
      with tf.variable_scope(self.name):
        predictions, loss = self._build_parallel(features, labels, params)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
          train_op = self._build_train_op(loss)
        else:
          train_op = None
      return predictions, loss, train_op

  def _build(self, features, labels, params):
    """Subclasses should implement this method. See the `model_fn` documentation
    in tf.contrib.learn.Estimator class for a more detailed explanation. This
    function should return a tuple of (predictions, loss)
    """
    raise NotImplementedError
