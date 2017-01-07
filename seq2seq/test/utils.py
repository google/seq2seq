"""Various testing utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

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
