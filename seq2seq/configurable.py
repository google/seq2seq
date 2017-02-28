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
Abstract base class for objects that are configurable using
a parameters dictionary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

import copy
import six

import tensorflow as tf

from seq2seq.graph_module import GraphModule

def _parse_params(params, default_params):
  # Cast parameters to correct types
  result = copy.deepcopy(default_params)
  for key, value in params.items():
    # If param is unknown, drop it to stay compatible with past versions
    if key not in default_params:
      tf.logging.warning("%s is not a valid model parameter, dropping", key)
      continue
    # Param is a dictionary
    if isinstance(value, dict):
      default_dict = default_params[key]
      if not isinstance(default_dict, dict):
        raise ValueError("%s should not be a dictionary", key)
      elif len(default_dict) == 0:
        # If the default is an empty dict we do not typecheck it
        # and assume it's done downstream
        pass
      else:
        value = _parse_params(value, default_params[key])
    result[key] = type(default_params[key])(value)
  return result


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
  def __init__(self, params):
    self._params = _parse_params(params, self.default_params())

  @property
  def params(self):
    return self._params

  @abc.abstractmethod
  def default_params(self):
    raise NotImplementedError
