#! /usr/bin/env python
""" Generates model predictions.
"""

import functools

import numpy as np
import tensorflow as tf

from seq2seq.inference import create_inference_graph, create_predictions_iter
from seq2seq.inference import unk_replace, get_unk_mapping

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
tf.flags.DEFINE_boolean("unk_replace", False,
                        """UNK token replacement strategy. If None (default)
                        do no replacement. "copy" copies source words based on
                        attention score. "probs" copies words based on attention
                        scores and a dictionary of probabilities.""")
tf.flags.DEFINE_string("unk_mapping", None,
                       """Used only if "unk_replace" is set to "props". This is
                       a conditional probability file such as the one generated
                       by fast_align.
                       Refer to the documentation for more details. """)

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def main(_argv):
  """Program entrypoint.
  """

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
  prediction_keys = set(
      ["predicted_tokens", "features.source_len", "features.source_tokens",
       "attention_scores"])

  # Optional UNK token replacement
  unk_replace_fn = None
  if FLAGS.unk_replace:
    if "attention_scores" not in predictions.keys():
      raise ValueError("""To perform UNK replacement you must use a model
                       class that outputs attention scores.""")
    prediction_keys.add("attention_scores")
    mapping = None
    if FLAGS.unk_mapping is not None:
      mapping = get_unk_mapping(FLAGS.unk_mapping)
    if FLAGS.unk_replace == "copy":
      unk_replace_fn = functools.partial(unk_replace, mapping=mapping)

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
    for predictions_dict in predictions_iter:
      # Convert to unicode
      predicted_tokens = predictions_dict["predicted_tokens"]
      predicted_tokens = np.char.decode(predicted_tokens.astype("S"), "utf-8")

      if FLAGS.beam_width is not None:
        # If we're using beam search we take the first beam
        predicted_tokens = predicted_tokens[:, 0]

      source_tokens = predictions_dict["features.source_tokens"]
      source_tokens = np.char.decode(source_tokens.astype("S"), "utf-8")

      if unk_replace_fn is not None:
        predicted_tokens = unk_replace_fn(
            source_tokens=source_tokens,
            predicted_tokens=predicted_tokens,
            attention_scores=predictions_dict["attention_scores"])

      print(" ".join(predicted_tokens).split("SEQUENCE_END")[0])

if __name__ == "__main__":
  tf.app.run()
