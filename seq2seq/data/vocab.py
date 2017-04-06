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
"""Vocabulary related functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow import gfile

SpecialVocab = collections.namedtuple("SpecialVocab",
                                      ["UNK", "SEQUENCE_START", "SEQUENCE_END"])


class VocabInfo(
    collections.namedtuple("VocbabInfo",
                           ["path", "vocab_size", "special_vocab"])):
  """Convenience structure for vocabulary information.
  """

  @property
  def total_size(self):
    """Returns size the the base vocabulary plus the size of extra vocabulary"""
    return self.vocab_size + len(self.special_vocab)


def get_vocab_info(vocab_path):
  """Creates a `VocabInfo` instance that contains the vocabulary size and
    the special vocabulary for the given file.

  Args:
    vocab_path: Path to a vocabulary file with one word per line.

  Returns:
    A VocabInfo tuple.
  """
  with gfile.GFile(vocab_path) as file:
    vocab_size = sum(1 for _ in file)
  special_vocab = get_special_vocab(vocab_size)
  return VocabInfo(vocab_path, vocab_size, special_vocab)


def get_special_vocab(vocabulary_size):
  """Returns the `SpecialVocab` instance for a given vocabulary size.
  """
  return SpecialVocab(*range(vocabulary_size, vocabulary_size + 3))


def create_vocabulary_lookup_table(filename, default_value=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
    """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    vocab = list(line.strip("\n") for line in file)
  vocab_size = len(vocab)

  has_counts = len(vocab[0].split("\t")) == 2
  if has_counts:
    vocab, counts = zip(*[_.split("\t") for _ in vocab])
    counts = [float(_) for _ in counts]
    vocab = list(vocab)
  else:
    counts = [-1. for _ in vocab]

  # Add special vocabulary items
  special_vocab = get_special_vocab(vocab_size)
  vocab += list(special_vocab._fields)
  vocab_size += len(special_vocab)
  counts += [-1. for _ in list(special_vocab._fields)]

  if default_value is None:
    default_value = special_vocab.UNK

  tf.logging.info("Creating vocabulary lookup table of size %d", vocab_size)

  vocab_tensor = tf.constant(vocab)
  count_tensor = tf.constant(counts, dtype=tf.float32)
  vocab_idx_tensor = tf.range(vocab_size, dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_idx_tensor, vocab_tensor, tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, vocab_idx_tensor, tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init,
                                                  default_value)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, count_tensor, tf.string, tf.float32)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size
