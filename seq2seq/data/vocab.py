"""Vocabulary related functions.
"""

import collections
import tensorflow as tf
from tensorflow.python.platform import gfile

SpecialVocab = collections.namedtuple("SpecialVocab",
                                      ["OOV", "SEQUENCE_START", "SEQUENCE_END"])


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


def create_vocabulary_lookup_table(filename, default_value=None, name=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: OOV tokens will be mapped to this id.
      If None, OOV tokens will be mapped to [vocab_size]
    name: Optional, a name for the operation

    Returns:
     A tuple (hash_table, reverse_hash_table, vocab_size). The vocab size
      does not include the OOV token.
    """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  with gfile.GFile(filename) as file:
    vocab_size = sum(1 for line in file)

  if default_value is None:
    default_value = vocab_size

  tf.logging.info("Creating vocabulary lookup table of size %d", vocab_size)

  table_init = tf.contrib.lookup.TextFileIdTableInitializer(
      filename, vocab_size=vocab_size)

  reverse_table_init = tf.contrib.lookup.TextFileInitializer(
      filename=filename,
      key_dtype=tf.int64,
      key_index=tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
      value_dtype=tf.string,
      value_index=tf.contrib.lookup.TextFileIndex.WHOLE_LINE,
      vocab_size=vocab_size)

  vocab_to_id_table = tf.contrib.lookup.HashTable(
      table_init, default_value, name=name)
  id_to_vocab_table = tf.contrib.lookup.HashTable(
      reverse_table_init, "UNK", name=name)

  return vocab_to_id_table, id_to_vocab_table, vocab_size
