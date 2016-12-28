""" Generates model predictions.
"""

import os
import itertools

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq import models
from seq2seq.data import vocab
from seq2seq.training import utils as training_utils
from seq2seq.training import HParamsParser

def load_model(vocab_source, vocab_target, model_class, model_dir, params=None):
  """Loads a model class from a given directory
  """
  # Load vocabulary
  source_vocab_info = vocab.get_vocab_info(vocab_source)
  target_vocab_info = vocab.get_vocab_info(vocab_target)

  # Find model class
  model_class = getattr(models, model_class)

  # Parse parameter and merge with defaults
  hparams = model_class.default_params()
  hparams_parser = HParamsParser(hparams)
  saved_hparams = training_utils.read_hparams(
      os.path.join(model_dir, "hparams.txt"))
  hparams = hparams_parser.parse(saved_hparams)

  if params is not None:
    hparams.update(params)

  # Create model instance
  model = model_class(
      source_vocab_info=source_vocab_info,
      target_vocab_info=target_vocab_info,
      params=hparams)

  return model


def print_translations(predictions_iter, vocab_path, use_beams=False):
  """Prints translations, one per line.
  """
  # Load the vocabulary in memory
  with gfile.GFile(vocab_path) as file:
    vocab_table = [l.strip() for l in file.readlines()]
  vocab_table += ["UNK", "SEQUENCE_START", "SEQUENCE_END"]

  # Print each predictions
  for prediction_dict in predictions_iter:
    token_ids = prediction_dict["predictions"]
    # If we're using beam search we take the first beam
    if use_beams:
      token_ids = token_ids[:, 0]
    tokens = [vocab_table[i] for i in token_ids]
    # Take sentence until SEQUENCE_END
    tokens = list(itertools.takewhile(lambda x: x != "SEQUENCE_END", tokens))
    sent = " ".join(tokens)
    print(sent)


def create_predictions_iter(predictions_dict, sess):
  """Runs prediciton fetches in a sessions and flattens batches as needed to
  return an iterator of predictions. Yield elements until an
  OutOfRangeError for the feeder queues occurs.

  Args:
    predictions_dict: The dictionary to be fetched. This will be passed
      to `session.run`. The first dimensions of each element in this
      dictionary is assumed to be the batch size.
    sess: The Session to use.

  Returns:
    An iterator of the same shape as predictions_dict, but with one
    element at a time and the batch dimension removed.
  """
  with tf.contrib.slim.queues.QueueRunners(sess):
    while True:
      try:
        predictions_ = sess.run(predictions_dict)
        batch_length = list(predictions_.values())[0].shape[0]
        for i in range(batch_length):
          yield {key: value[i] for key, value in predictions_.items()}
      except tf.errors.OutOfRangeError:
        break
