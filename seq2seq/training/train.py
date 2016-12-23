"""Main script to run training and evaluation of models.
"""

#! /usr/bin/env python

import os
import tempfile
from seq2seq import models
from seq2seq.data import data_utils, vocab
from seq2seq.training import HParamsParser
from seq2seq.training import utils as training_utils
from seq2seq.training import hooks
from seq2seq.training import metrics

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.platform import gfile

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
                        Default is 5.""")
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

tf.logging.set_verbosity(tf.logging.INFO)


def create_experiment(output_dir):
  """
  Creates a new Experiment instance.

  Args:
    output_dir: Output directory for model checkpoints and summaries.
  """

  # Load vocabulary info
  source_vocab_info = vocab.get_vocab_info(FLAGS.vocab_source)
  target_vocab_info = vocab.get_vocab_info(FLAGS.vocab_target)

  # Create data providers
  train_data_provider = \
    lambda: data_utils.make_parallel_data_provider(
        [FLAGS.train_source], [FLAGS.train_target], shuffle=True)
  dev_data_provider = \
    lambda: data_utils.make_parallel_data_provider(
        [FLAGS.dev_source], [FLAGS.dev_target], num_epochs=1)

  # Find model class
  model_class = getattr(models, FLAGS.model)

  # Parse parameter and merge with defaults
  hparams = model_class.default_params()
  if FLAGS.hparams is not None:
    hparams = HParamsParser(hparams).parse(FLAGS.hparams)

  # Print hyperparameter values
  tf.logging.info("Model Hyperparameters")
  tf.logging.info("=" * 50)
  for param, value in sorted(hparams.items()):
    tf.logging.info("%s=%s", param, value)
  tf.logging.info("=" * 50)
  # Write hparams to file
  gfile.MakeDirs(output_dir)
  hparams_path = os.path.join(output_dir, "hparams.txt")
  training_utils.write_hparams(hparams, hparams_path)

  # Create model
  model = model_class(
      source_vocab_info=source_vocab_info,
      target_vocab_info=target_vocab_info,
      params=hparams)
  featurizer = model.create_featurizer()

  bucket_boundaries = None
  if FLAGS.buckets:
    bucket_boundaries = list(map(int, FLAGS.buckets.split(",")))

  # Create input functions
  train_input_fn = training_utils.create_input_fn(
      train_data_provider,
      featurizer,
      FLAGS.batch_size,
      bucket_boundaries=bucket_boundaries)
  eval_input_fn = training_utils.create_input_fn(dev_data_provider, featurizer,
                                                 FLAGS.batch_size)

  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    return model(features, labels, params, mode)

  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours
  )
  estimator = tf.contrib.learn.estimator.Estimator(
      model_fn=model_fn, model_dir=output_dir, config=config)

  # Create training Hooks
  model_analysis_hook = hooks.PrintModelAnalysisHook(
      filename=os.path.join(estimator.model_dir, "model_analysis.txt"))
  sample_file = os.path.join(output_dir, "samples.txt")
  train_sample_hook = hooks.TrainSampleHook(
      every_n_steps=FLAGS.sample_every_n_steps, file=sample_file)
  metadata_hook = hooks.MetadataCaptureHook(
      output_dir=os.path.join(estimator.model_dir, "metadata"), step=10)
  tokens_per_sec_counter = hooks.TokensPerSecondCounter(every_n_steps=100)
  train_monitors = [
      model_analysis_hook, train_sample_hook, metadata_hook,
      tokens_per_sec_counter
  ]

  # Metrics
  eval_metrics = {"log_perplexity": metrics.streaming_log_perplexity()}

  experiment = tf.contrib.learn.experiment.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      min_eval_frequency=FLAGS.eval_every_n_steps,
      train_steps=FLAGS.train_steps,
      eval_steps=None,
      eval_metrics=eval_metrics,
      train_monitors=train_monitors)

  return experiment


def main(_argv):
  """The entrypoint for the script"""
  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()
  learn_runner.run(experiment_fn=create_experiment,
                   output_dir=FLAGS.output_dir,
                   schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.app.run()
