#! /usr/bin/env python
""" Generates model predictions.
"""

import os
import itertools

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq import models
from seq2seq.data import data_utils, vocab
from seq2seq.training import utils as training_utils
from seq2seq.training import HParamsParser

tf.flags.DEFINE_string("source", None, "path to source training data")
tf.flags.DEFINE_string("vocab_source", None, "Path to source vocabulary file")
tf.flags.DEFINE_string("vocab_target", None, "Path to target vocabulary file")
tf.flags.DEFINE_string("model", "AttentionSeq2Seq", "model class")
tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def load_model(vocab_source, vocab_target, model_class, model_dir):
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

  # Create model instance
  model = model_class(
      source_vocab_info=source_vocab_info,
      target_vocab_info=target_vocab_info,
      params=hparams)

  return model


def create_estimator(model, model_dir):
  """Creates an estimator for a  model function and checkpoint directory.
  """

  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    return model(features, labels, params, mode)

  return tf.contrib.learn.estimator.Estimator(
      model_fn=model_fn, model_dir=model_dir)


def print_translations(predictions_iter, vocab_path):
  """Prints translations, one per line.
  """
  # Load the vocabulary in memory
  with gfile.GFile(vocab_path) as file:
    vocab_table = [l.strip() for l in file.readlines()]
  vocab_table += ["OOV", "SEQUENCE_START", "SEQUENCE_END"]

  # Print each predictions
  for prediction_dict in predictions_iter:
    tokens = [vocab_table[i] for i in prediction_dict["predictions"]]
    # Take sentence until SEQUENCE_END
    tokens = list(itertools.takewhile(lambda x: x != "SEQUENCE_END", tokens))
    sent = " ".join(tokens)
    print(sent)


def create_input_fn(model, input_file, batch_size):
  """Creates an infernece input function for the given file.
  """
  # TODO: We use the input_file as both source and target here, but
  # the target is ignored during inference. We need to pass the target anyway
  # because the featurizer expects it as an argument. Should fix that.
  data_provider = lambda: data_utils.make_parallel_data_provider(
      [input_file], [input_file], shuffle=False, num_epochs=1)

  input_fn = training_utils.create_input_fn(
      data_provider_fn=data_provider,
      featurizer_fn=model.create_featurizer(),
      batch_size=batch_size,
      allow_smaller_final_batch=True)
  return input_fn


def main(_argv):
  """Program entrypoint.
  """
  model = load_model(
      vocab_source=FLAGS.vocab_source,
      vocab_target=FLAGS.vocab_target,
      model_class=FLAGS.model,
      model_dir=FLAGS.model_dir)
  estimator = create_estimator(model, FLAGS.model_dir)
  input_fn = create_input_fn(model, FLAGS.source, FLAGS.batch_size)
  predictions_iter = estimator.predict(input_fn=input_fn)
  print_translations(predictions_iter, FLAGS.vocab_target)


if __name__ == "__main__":
  tf.app.run()
