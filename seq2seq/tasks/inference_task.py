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
  def prediction_keys(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def process_batch(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_model(self, mode):
    raise NotImplementedError()
