#! /usr/bin/env python
""" Generates attention score visualizations.
"""

import os

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
from matplotlib import pyplot as plt

from seq2seq import graph_utils
from seq2seq.inference import create_inference_graph, create_predictions_iter

tf.flags.DEFINE_string("source", None, "path to source training data")
tf.flags.DEFINE_string("vocab_source", None, "Path to source vocabulary file")
tf.flags.DEFINE_string("vocab_target", None, "Path to target vocabulary file")
tf.flags.DEFINE_string("model", "AttentionSeq2Seq", "model class")
tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")
tf.flags.DEFINE_integer("beam_width", None,
                        "Use beam search with this beam width for decoding")

tf.flags.DEFINE_string("output_dir", None,
                       "Write all attention plots to this directory")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def create_figure(predictions_dict):
  """Creates an returns a new figure that visualizes
  attention scors for for a single model predictions.
  """

  # Find out how long the predicted sequence is
  target_words = list(predictions_dict["predicted_tokens"])
  target_words = [_.decode("utf-8") for _ in target_words]

  prediction_len = next(
      ((i + 1) for i, _ in enumerate(target_words) if _ == "SEQUENCE_END"),
      None)
  # Get source words
  source_len = predictions_dict["features.source_len"]
  source_words = predictions_dict["features.source_tokens"][:source_len]
  source_words = [_.decode("utf-8") for _ in source_words]

  # Plot
  fig = plt.figure(figsize=(8, 8))
  plt.imshow(
      X=predictions_dict["attention_scores"][:prediction_len, :],
      interpolation="nearest",
      cmap=plt.cm.Blues)
  plt.xticks(np.arange(source_len), source_words, rotation=45)
  plt.yticks(np.arange(prediction_len), target_words, rotation=-45)
  fig.tight_layout()

  return fig


def main(_argv):
  """Program entrypoint.
  """

  gfile.MakeDirs(FLAGS.output_dir)

  predictions, _, _ = create_inference_graph(
      model_class=FLAGS.model,
      model_dir=FLAGS.model_dir,
      vocab_source=FLAGS.vocab_source,
      vocab_target=FLAGS.vocab_target,
      input_file=FLAGS.source,
      batch_size=FLAGS.batch_size,
      beam_width=FLAGS.beam_width
  )

  # Filter fetched predictions to save memory
  prediction_keys = set(["predicted_tokens", "attention_scores",
                         "features.source_len", "features.source_tokens"])
  predictions = {k: v for k, v in predictions.items() if k in prediction_keys}

  saver = tf.train.Saver()

  checkpoint_path = FLAGS.checkpoint_path
  if not checkpoint_path:
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.initialize_all_tables())

    # Restore checkpoint
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Restored model from %s", checkpoint_path)

    # Output predictions
    predictions_iter = create_predictions_iter(predictions, sess)

    for idx, predictions_dict in enumerate(predictions_iter):
      output_path = os.path.join(FLAGS.output_dir, "{:05d}.png".format(idx))
      create_figure(predictions_dict)
      plt.savefig(output_path)
      tf.logging.info("Wrote %s", output_path)

if __name__ == "__main__":
  tf.app.run()
