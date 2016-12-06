#! /usr/bin/env python

import os
import tensorflow as tf
import seq2seq

# Data
tf.flags.DEFINE_string("data_train", None, "path to training data TFRecords")
tf.flags.DEFINE_string("data_dev", None, "path to dev data TFRecords")
tf.flags.DEFINE_string("vocab_source", None, "Path to source vocabulary file")
tf.flags.DEFINE_string("vocab_target", None, "Path to target vocabulary file")

tf.flags.DEFINE_integer("batch_size", 16, "the train/dev batch size")
tf.flags.DEFINE_string("hparams", None, "overwrite hyperparameter values")
tf.flags.DEFINE_string("model", "BasicSeq2Seq", "model class")
tf.flags.DEFINE_string("model_dir", None, "directory to write to")
tf.flags.DEFINE_integer("save_checkpoints_secs", 300, "save checkpoint every N seconds")

tf.flags.DEFINE_integer("train_steps", None, "maximum number of training steps")
tf.flags.DEFINE_integer("eval_steps", 100, "maxmum number of eval steps")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000, "evaluate after this many training steps")

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def main(_argv):
  # Load vocabulary info
  source_vocab_info = seq2seq.inputs.get_vocab_info(FLAGS.vocab_source)
  target_vocab_info = seq2seq.inputs.get_vocab_info(FLAGS.vocab_target)

  # Create data providers
  train_data_provider = lambda: seq2seq.inputs.make_data_provider([FLAGS.data_train])
  dev_data_provider = lambda: seq2seq.inputs.make_data_provider([FLAGS.data_dev])

  # Create input functions for training and eval
  # TOOD: Move featurizer into model. The model should know how to featurize the data
  # for itself
  featurizer = seq2seq.training.featurizers.Seq2SeqFeaturizer(
    source_vocab_info, target_vocab_info)
  train_input_fn = seq2seq.training.utils.create_input_fn(
    train_data_provider, featurizer, FLAGS.batch_size)
  eval_input_fn = seq2seq.training.utils.create_input_fn(
    dev_data_provider, featurizer, FLAGS.batch_size)

  model_class = getattr(seq2seq.models, FLAGS.model)
  model = model_class(
      source_vocab_info=source_vocab_info,
      target_vocab_info=target_vocab_info,
      params=model_class.default_params())

  def model_fn(features, labels, params, mode):
    return model(features, labels, params, mode)

  estimator = tf.contrib.learn.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir)

  # Create training Hooks
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=eval_input_fn, eval_steps=FLAGS.eval_steps, every_n_steps=FLAGS.eval_every_n_steps)
  model_analysis_hook = seq2seq.training.hooks.PrintModelAnalysisHook(
    filename=os.path.join(estimator.model_dir, "model_analysis.txt"))
  train_sample_hook = seq2seq.training.hooks.TrainSampleHook(every_n_secs=60)

  estimator.fit(
    input_fn=train_input_fn,
    steps=FLAGS.train_steps,
    monitors=[model_analysis_hook, train_sample_hook, validation_monitor])

if __name__ == "__main__":
  tf.app.run()
