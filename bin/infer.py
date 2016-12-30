#! /usr/bin/env python
""" Generates model predictions.
"""

import tensorflow as tf

from seq2seq.inference import create_inference_graph, create_predictions_iter
from seq2seq.inference import print_translations

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
    print_translations(
        predictions_iter=predictions_iter,
        use_beams=(FLAGS.beam_width is not None))

if __name__ == "__main__":
  tf.app.run()
