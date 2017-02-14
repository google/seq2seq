"""Various testing utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf

def create_temp_parallel_data(sources, targets):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """
  file_source = tempfile.NamedTemporaryFile()
  file_target = tempfile.NamedTemporaryFile()
  file_source.write("\n".join(sources).encode("utf-8"))
  file_source.flush()
  file_target.write("\n".join(targets).encode("utf-8"))
  file_target.flush()
  return file_source, file_target


def create_temp_tfrecords(sources, targets):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """

  output_file = tempfile.NamedTemporaryFile()
  writer = tf.python_io.TFRecordWriter(output_file.name)
  for source, target in zip(sources, targets):
    ex = tf.train.Example()
    ex.features.feature["source"].bytes_list.value.extend(
        [source.encode("utf-8")])
    ex.features.feature["target"].bytes_list.value.extend(
        [target.encode("utf-8")])
    writer.write(ex.SerializeToString())
  writer.close()

  return output_file


def create_temporary_vocab_file(words):
  """
  Creates a temporary vocabulary file.

  Args:
    words: List of words in the vocabulary

  Returns:
    A temporary file object with one word per line
  """
  vocab_file = tempfile.NamedTemporaryFile()
  for token in words:
    vocab_file.write((token + "\n").encode("utf-8"))
  vocab_file.flush()
  return vocab_file
