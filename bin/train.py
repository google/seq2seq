#! /usr/bin/env python

"""Main script to run training and evaluation of models.
"""

import functools
import os
import tempfile
from seq2seq import models
from seq2seq.data import data_utils, vocab
from seq2seq.training import HParamsParser
from seq2seq.training import utils as training_utils
from seq2seq.training import metrics

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config

# Input Data
tf.flags.DEFINE_string("train_source", None, "path to source training data")
tf.flags.DEFINE_string("train_target", None, "path to target training data")
tf.flags.DEFINE_string("dev_source", None, "path to source dev data")
tf.flags.DEFINE_string("dev_target", None, "path to target dev data")
tf.flags.DEFINE_string("vocab_source", None, "Path to source vocabulary file")
tf.flags.DEFINE_string("vocab_target", None, "Path to target vocabulary file")

# Model Configuration
tf.flags.DEFINE_string("buckets", None,
                       """A comma-separated list of sequence lenght buckets,
                       e.g. 10,20,30""")
tf.flags.DEFINE_integer("batch_size", 16, "the train/dev batch size")
tf.flags.DEFINE_string("hparams", None, "overwrite hyperparameter values")
tf.flags.DEFINE_string("model", "BasicSeq2Seq", "model class")
tf.flags.DEFINE_string("output_dir", None, "directory to write to")

# Training parameters
tf.flags.DEFINE_string("schedule", None,
                       """Estimator function to call, defaults to
                       train_and_evaluate for local run""")
tf.flags.DEFINE_integer("train_steps", None, "maximum number of training steps")
tf.flags.DEFINE_integer("train_epochs", None,
                        """Maximum number of training epochs. Defaults to None,
                        which means train forever.""")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "evaluate after this many training steps")
tf.flags.DEFINE_integer("sample_every_n_steps", 500,
                        "sample training predictions every N steps")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", 600,
                        """Save checkpoints every this many seconds.
                        Can not be specified with save_checkpoints_steps.
                        Default is 600.""")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        """Save checkpoints every this many steps.
                        Can not be specified with save_checkpoints_secs.""")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        """Maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted.
                        If None or 0, all checkpoint files are kept.
                        Defaults to 5 (that is, the 5 most recent checkpoint
                        files are kept.)""")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                        """Number of hours between each checkpoint to be saved.
                        Default is 4.""")

FLAGS = tf.flags.FLAGS

def create_experiment(output_dir):
  """
  Creates a new Experiment instance.

  Args:
    output_dir: Output directory for model checkpoints and summaries.
  """

  # Load vocabulary info
  source_vocab_info = vocab.get_vocab_info(FLAGS.vocab_source)
  target_vocab_info = vocab.get_vocab_info(FLAGS.vocab_target)

  # Find model class
  model_class = getattr(models, FLAGS.model)

  # Parse parameter and merge with defaults
  hparams = model_class.default_params()
  if FLAGS.hparams is not None:
    hparams = HParamsParser(hparams).parse(FLAGS.hparams)

  # Print and save hparams
  training_utils.print_hparams(hparams)
  training_utils.write_hparams(
      hparams, os.path.join(output_dir, "hparams.txt"))

  # Create model
  model = model_class(
      source_vocab_info=source_vocab_info,
      target_vocab_info=target_vocab_info,
      params=hparams)

  bucket_boundaries = None
  if FLAGS.buckets:
    bucket_boundaries = list(map(int, FLAGS.buckets.split(",")))

  # Create training input function
  train_input_fn = training_utils.create_input_fn(
      data_provider_fn=functools.partial(
          data_utils.make_parallel_data_provider,
          data_sources_source=FLAGS.train_source,
          data_sources_target=FLAGS.train_target,
          shuffle=True,
          num_epochs=FLAGS.train_epochs),
      batch_size=FLAGS.batch_size,
      bucket_boundaries=bucket_boundaries)

  # Create eval input function
  eval_input_fn = training_utils.create_input_fn(
      data_provider_fn=functools.partial(
          data_utils.make_parallel_data_provider,
          data_sources_source=FLAGS.dev_source,
          data_sources_target=FLAGS.dev_target,
          shuffle=False,
          num_epochs=1),
      batch_size=FLAGS.batch_size)

  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    result = model(features, labels, params, mode)

    # Create a custom saver
    # This is necessary to support "keep_checkpoint_every_n_hours"
    # which is currently ignored by Tensorflow, see
    # https://github.com/tensorflow/tensorflow/issues/6549
    saver = tf.train.Saver(
        sharded=True,
        max_to_keep=FLAGS.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        defer_build=(mode == tf.contrib.learn.ModeKeys.TRAIN))
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

    return result

  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours
  )

  estimator = tf.contrib.learn.estimator.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=config)

  train_hooks = training_utils.create_default_training_hooks(
      output_dir=output_dir,
      sample_frequency=FLAGS.sample_every_n_steps)

  eval_metrics = {
      "log_perplexity": metrics.streaming_log_perplexity()
  }

  experiment = tf.contrib.learn.experiment.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      min_eval_frequency=FLAGS.eval_every_n_steps,
      train_steps=FLAGS.train_steps,
      eval_steps=None,
      eval_metrics=eval_metrics,
      train_monitors=train_hooks)

  return experiment


def main(_argv):
  """The entrypoint for the script"""
  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()

  learn_runner.run(
      experiment_fn=create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
