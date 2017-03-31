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
from seq2seq import global_vars


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

  def _create_optimizer(self):
    """Creates the optimizer"""
    name = self.params["optimizer.name"]
    optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
        learning_rate=self.params["optimizer.learning_rate"],
        **self.params["optimizer.params"])

    # Optionally wrap with SyncReplicasOptimizer
    if self.params["optimizer.sync_replicas"] > 0:
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=self.params[
              "optimizer.sync_replicas_to_aggregate"],
          total_num_replicas=self.params["optimizer.sync_replicas"])
      # This is really ugly, but we need to do this to make the optimizer
      # accessible outside of the model.
      global_vars.SYNC_REPLICAS_OPTIMIZER = optimizer

    return optimizer

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

    optimizer = self._create_optimizer()
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=self.params["optimizer.learning_rate"],
        learning_rate_decay_fn=learning_rate_decay_fn,
        clip_gradients=self._clip_gradients,
        optimizer=optimizer,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return train_op

  @staticmethod
  def default_params():
    """Returns a dictionary of default parameters for this model."""
    return {
        "optimizer.name": "Adam",
        "optimizer.learning_rate": 1e-4,
        "optimizer.params": {}, # Arbitrary parameters for the optimizer
        "optimizer.lr_decay_type": "",
        "optimizer.lr_decay_steps": 100,
        "optimizer.lr_decay_rate": 0.99,
        "optimizer.lr_start_decay_at": 0,
        "optimizer.lr_stop_decay_at": tf.int32.max,
        "optimizer.lr_min_learning_rate": 1e-12,
        "optimizer.lr_staircase": False,
        "optimizer.clip_gradients": 5.0,
        "optimizer.sync_replicas": 0,
        "optimizer.sync_replicas_to_aggregate": 0,
    }

  def batch_size(self, features, labels):
    """Returns the batch size for a batch of examples"""
    raise NotImplementedError()

  def __call__(self, features, labels, params):
    """Creates the model graph. See the model_fn documentation in
    tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    with tf.variable_scope("model"):
      with tf.variable_scope(self.name):
        return self._build(features, labels, params)

  def _build(self, features, labels, params):
    """Subclasses should implement this method. See the `model_fn` documentation
    in tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    raise NotImplementedError
