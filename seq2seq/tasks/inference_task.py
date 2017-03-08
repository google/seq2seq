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

"""
Abstract base class for inference tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import abc
from pydoc import locate

import six

from seq2seq import models
from seq2seq.configurable import Configurable, _deep_merge_dict

@six.add_metaclass(abc.ABCMeta)
class InferenceTask(Configurable):
  """
  Abstract base class for inference tasks. Defines the logic used make
  predictions for a specific type of task.

  Params:
    model_class: The model class to instantiate. If undefined,
      re-uses the class used during training.
    model_params: Model hyperparameters. Specified hyperparameters will
      overwrite those used during training.

  Args:
    params: See Params above.
    train_options: A `TrainOptions` instance.
  """
  def __init__(self, params, train_options):
    # Merge model parameters with those used during training
    if "model_params" not in params:
      params["model_params"] = {}
    params["model_params"] = _deep_merge_dict(
        copy.deepcopy(train_options.task_params["model_params"]),
        params["model_params"])

    # If the model class is specified, use it, otherwise use the
    # model class from training
    if "model_class" not in params:
      params["model_class"] = train_options.task_params["model_class"]

    super(InferenceTask, self).__init__(params, None)
    self._train_options = train_options
    self._model_cls = locate(self.params["model_class"]) or \
      getattr(models, self.params["model_class"])

  @staticmethod
  def default_params():
    return {
        "model_class": None,
        "model_params": {},
    }

  @abc.abstractmethod
  def create_model(self):
    """Creates a model instance in inference mode.

    Returns:
      A model instance.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def prediction_keys(self):
    """Defines which predictions tensors should be fetched from the model.

    Returns:
      A set of strings, each corresponding to an item in the model
      predictions.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def begin(self):
    """Initializes the task. This method will be called after the graph is
    created and before the first call to `process_batch.`
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def process_batch(self, idx, predictions_dict):
    """Processes a single batch of predictions. This is where most of the
    inference logic resides.

    Args:
      idx: 0-based index of the batch to be processed
      predictions_dict: A dictionary of numpy arrays containing the model
        predictions. Keys correspond to the items returned by `prediction_keys`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def end(self):
    """Finishes the task. This method will be called after iteration
    through the data is complete and all batches have been processed.
    """
    raise NotImplementedError()
