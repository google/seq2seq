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
"""Miscellaneous utility function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def templatemethod(name_):
  """This decorator wraps a method with `tf.make_template`. For example,

  @templatemethod
  def my_method():
    # Create variables
  """

  def template_decorator(func):
    """Inner decorator function"""

    def func_wrapper(*args, **kwargs):
      """Inner wrapper function"""
      templated_func = tf.make_template(name_, func)
      return templated_func(*args, **kwargs)

    return func_wrapper

  return template_decorator


def add_dict_to_collection(dict_, collection_name):
  """Adds a dictionary to a graph collection.

  Args:
    dict_: A dictionary of string keys to tensor values
    collection_name: The name of the collection to add the dictionary to
  """
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)


def get_dict_from_collection(collection_name):
  """Gets a dictionary from a graph collection.

  Args:
    collection_name: A collection name to read a dictionary from

  Returns:
    A dictionary with string keys and tensor values
  """
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  keys = tf.get_collection(key_collection)
  values = tf.get_collection(value_collection)
  return dict(zip(keys, values))
