#! /usr/bin/env python

"""Main script to run training and evaluation of models.
"""

import functools
import os
import tempfile
import yaml

from seq2seq import models
from seq2seq.data import data_utils, vocab
from seq2seq.training import HParamsParser
from seq2seq.training import utils as training_utils
from seq2seq.training import metrics

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.platform import gfile


# Input Data
tf.flags.DEFINE_string("train_source", None,
                       """Path to the training data source sentences. A raw
                       text files with tokens separated by spaces.""")
tf.flags.DEFINE_string("train_target", None,
                       """Path to the training data target sentences. A raw
                       text files with tokens separated by spaces.""")
tf.flags.DEFINE_string("dev_source", None,
                       """Path to the development data source sentences.
                       Same format as training data.""")
tf.flags.DEFINE_string("dev_target", None,
                       """Path to the development data target sentences.
                       Same format as training data.""")
tf.flags.DEFINE_string("vocab_source", None,
                       """Path to the source vocabulary.
                       A raw text file with one word per line.""")
tf.flags.DEFINE_string("vocab_target", None,
                       """Path to the target vocabulary.
                       A raw text file with one word per line.""")
tf.flags.DEFINE_string("delimiter", " ",
                       """Split input files into tokens on this delimiter.
                      Defaults to " " (space).""")
tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

# Model Configuration
tf.flags.DEFINE_string("model", "AttentionSeq2Seq",
                       """The model class to use. Refer to the documentation
                       for all available models.""")
tf.flags.DEFINE_string("buckets", None,
                       """Buckets input sequences according to these length.
                       A comma-separated list of sequence length buckets, e.g.
                       "10,20,30" would result in 4 buckets:
                       <10, 10-20, 20-30, >30. None disabled bucketing. """)
tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("hparams", None,
                       """A comma-separated list of hyeperparameter values that
                       overwrite the model defaults, e.g.
                       "optimizer.name=Adam,optimization.learning_rate=0.1".
                       Refer to the documentation for a detailed list of
                       available hyperparameters.""")
tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

# Training parameters
tf.flags.DEFINE_string("schedule", None,
                       """Estimator function to call, defaults to
                       train_and_evaluate for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
tf.flags.DEFINE_integer("train_epochs", None,
                        """Maximum number of training epochs over the data.
                         If None, train forever.""")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")
tf.flags.DEFINE_integer("sample_every_n_steps", 500,
                        """Sample and print sequence predictions every N steps
                        during training.""")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", 600,
                        """Save checkpoints every this many seconds.
                        Can not be specified with save_checkpoints_steps.""")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        """Save checkpoints every this many steps.
                        Can not be specified with save_checkpoints_secs.""")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        """Maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted.
                        If None or 0, all checkpoint files are kept.""")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                        """In addition to keeping the most recent checkpoint
                        files, keep one checkpoint file for every N hours of
                        training.""")

FLAGS = tf.flags.FLAGS

def create_experiment(output_dir):
  """
  Creates a new Experiment instance.

  Args:
    output_dir: Output directory for model checkpoints and summaries.
  """

  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours
  )

  # Load vocabulary info
  source_vocab_info = vocab.get_vocab_info(FLAGS.vocab_source)
  target_vocab_info = vocab.get_vocab_info(FLAGS.vocab_target)

  # Find model class
  model_class = getattr(models, FLAGS.model)

  # Parse parameter and merge with defaults
  hparams = model_class.default_params()
  if FLAGS.hparams is not None and isinstance(FLAGS.hparams, str):
    hparams = HParamsParser(hparams).parse(FLAGS.hparams)
  elif isinstance(FLAGS.hparams, dict):
    hparams.update(FLAGS.hparams)

  # Print hparams
  training_utils.print_hparams(hparams)

  # One the main worker, save training options and vocabulary
  if config.is_chief:
     # Copy vocabulary to output directory
    gfile.MakeDirs(output_dir)
    source_vocab_path = os.path.join(output_dir, "vocab_source")
    gfile.Copy(FLAGS.vocab_source, source_vocab_path, overwrite=True)
    target_vocab_path = os.path.join(output_dir, "vocab_target")
    gfile.Copy(FLAGS.vocab_target, target_vocab_path, overwrite=True)
    # Save train options
    train_options = training_utils.TrainOptions(
        hparams=hparams,
        model_class=FLAGS.model,
        source_vocab_path=source_vocab_path,
        target_vocab_path=target_vocab_path)
    train_options.dump(output_dir)

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
          num_epochs=FLAGS.train_epochs,
          delimiter=FLAGS.delimiter),
      batch_size=FLAGS.batch_size,
      bucket_boundaries=bucket_boundaries)

  # Create eval input function
  eval_input_fn = training_utils.create_input_fn(
      data_provider_fn=functools.partial(
          data_utils.make_parallel_data_provider,
          data_sources_source=FLAGS.dev_source,
          data_sources_target=FLAGS.dev_target,
          shuffle=False,
          num_epochs=1,
          delimiter=FLAGS.delimiter),
      batch_size=FLAGS.batch_size)

  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    return model(features, labels, params, mode)

  estimator = tf.contrib.learn.estimator.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=config)

  train_hooks = training_utils.create_default_training_hooks(
      estimator=estimator,
      sample_frequency=FLAGS.sample_every_n_steps,
      delimiter=FLAGS.delimiter)

  eval_metrics = {
      "log_perplexity": metrics.streaming_log_perplexity(),
      "bleu": metrics.make_bleu_metric_spec(),
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

  # Load flags from config file
  if FLAGS.config_path:
    with gfile.GFile(FLAGS.config_path) as config_file:
      config_flags = yaml.load(config_file)
      for flag_key, flag_value in config_flags.items():
        setattr(FLAGS, flag_key, flag_value)

  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()

  learn_runner.run(
      experiment_fn=create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
