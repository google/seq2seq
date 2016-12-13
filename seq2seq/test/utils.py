"""Various testing utilities
"""

import tempfile
from seq2seq.scripts import generate_examples
from seq2seq.data import data_utils


def create_temp_tfrecords(source, target):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source words
    target: List of target words

  Returns:
    A temporary file object
  """
  file = tempfile.NamedTemporaryFile()
  ex = generate_examples.build_example(pair_id=0, source=source, target=target)
  generate_examples.write_tfrecords([ex], file.name)
  return file

def create_temp_parallel_data(sources, targets):
  """
  Creates a temporary TFRecords file.

  Args:
    source: List of source sentences
    target: List of target sentences

  Returns:
    A tuple (sources_file, targets_file).
  """
  file_source = tempfile.NamedTemporaryFile("w")
  file_target = tempfile.NamedTemporaryFile("w")
  file_source.write("\n".join(sources))
  file_source.flush()
  file_target.write("\n".join(targets))
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
  vocab_file = tempfile.NamedTemporaryFile("w")
  for token in words:
    vocab_file.write(token + "\n")
  vocab_file.flush()
  return vocab_file

