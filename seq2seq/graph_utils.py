"""Miscellaneous utility function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
  """Adds a dictionary to a graph collection.

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
