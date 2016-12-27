"""A decoder that splits a string into tokens and returns the
individual tokens and the length.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder


class SplitTokensDecoder(data_decoder.DataDecoder):
  """A DataProvider that splits a string tensor into individual tokens and
  returns the tokens and the length.
  Optionally prepends or appends special tokens.

  Args:
    delimiter: Delimiter to split on. Must be a single character.
    tokens_feature_name: A descriptive feature name for the token values
    length_feature_name: A descriptive feature name for the length value
  """

  def __init__(self,
               delimiter=" ",
               tokens_feature_name="tokens",
               length_feature_name="length",
               prepend_token=None,
               append_token=None):
    self.delimiter = delimiter
    self.tokens_feature_name = tokens_feature_name
    self.length_feature_name = length_feature_name
    self.prepend_token = prepend_token
    self.append_token = append_token

  def decode(self, data, items):
    decoded_items = {}

    # Split tokens
    tokens = tf.string_split([data], delimiter=self.delimiter).values

    # Optionally prepend a special token
    if self.prepend_token is not None:
      tokens = tf.concat_v2([[self.prepend_token], tokens], 0)

    # Optionally append a special token
    if self.append_token is not None:
      tokens = tf.concat_v2([tokens, [self.append_token]], 0)

    decoded_items[self.length_feature_name] = tf.size(tokens)
    decoded_items[self.tokens_feature_name] = tokens
    return [decoded_items[_] for _ in items]

  def list_items(self):
    return [self.tokens_feature_name, self.length_feature_name]
