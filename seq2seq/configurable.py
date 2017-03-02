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

import copy
import six

import tensorflow as tf

def _parse_params(params, default_params):
  """Parses parameter values to the types defined by the default parameters.
  Default parameters are used for missing values.
  """
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
      if default_dict:
        value = _parse_params(value, default_dict)
      else:
        # If the default is an empty dict we do not typecheck it
        # and assume it's done downstream
        pass
    result[key] = type(default_params[key])(value)
  return result


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
  """Interface for all classes that are configurable
  via a parameters dictionary.

  Args:
    params: A dictionary of parameters.
    mode: A value in tf.contrib.learn.ModeKeys
  """
  def __init__(self, params, mode):
    self._params = _parse_params(params, self.default_params())
    self._mode = mode

  @property
  def mode(self):
    """Returns a value in tf.contrib.learn.ModeKeys.
    """
    return self._mode

  @property
  def params(self):
    """Returns a dictionary of parsed parameters.
    """
    return self._params

  @abc.abstractmethod
  def default_params(self):
    """Returns a dictionary of default parameters. The default parameters
    are used to define the expected type of passed parameters. Missing
    parameter values are replaced with the defaults returned by this method.
    """
    raise NotImplementedError
