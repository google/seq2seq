"""Various testing utilities
"""

import tempfile
from seq2seq.scripts import generate_examples
from seq2seq import inputs


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


def create_next_input_fn_for_test(source, target):
  """
  Creates a input function that reads a given source and target.

  Args:
    source: List of source words
    target: List of target words

  Returns:
    A function that reads from a temporary file
  """
  file = create_temp_tfrecords(source, target)

  def next_input_fn():
    """
    The input function that is returned.
    """
    data_provider = inputs.make_data_provider([file.name], num_epochs=None)
    return inputs.read_from_data_provider(data_provider)

  return next_input_fn
