#! /usr/bin/env python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import yaml

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow import gfile

from seq2seq import models
from seq2seq.contrib.experiment import Experiment as PatchedExperiment
from seq2seq.configurable import _maybe_load_yaml, _create_from_dict
from seq2seq.configurable import _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.metrics import metric_specs
from seq2seq.training import hooks
from seq2seq.training import utils as training_utils

tf.flags.DEFINE_string("config_paths", "",
                       """Path to a YAML configuration files defining FLAG
                       values. Multiple files can be separated by commas.
                       Files are merged recursively. Setting a key in these
                       files is equivalent to setting the FLAG value with
                       the same name.""")
tf.flags.DEFINE_string("hooks", "[]",
                       """YAML configuration string for the
                       training hooks to use.""")
tf.flags.DEFINE_string("metrics", "[]",
                       """YAML configuration string for the
                       training metrics to use.""")
tf.flags.DEFINE_string("model", "",
                       """Name of the model class.
                       Can be either a fully-qualified name, or the name
                       of a class defined in `seq2seq.models`.""")
tf.flags.DEFINE_string("model_params", "{}",
                       """YAML configuration string for the model
                       parameters.""")

tf.flags.DEFINE_string("input_pipeline_train", "{}",
                       """YAML configuration string for the training
                       data input pipeline.""")
tf.flags.DEFINE_string("input_pipeline_dev", "{}",
                       """YAML configuration string for the development
                       data input pipeline.""")

tf.flags.DEFINE_string("buckets", None,
                       """Buckets input sequences according to these length.
                       A comma-separated list of sequence length buckets, e.g.
                       "10,20,30" would result in 4 buckets:
                       <10, 10-20, 20-30, >30. None disabled bucketing. """)
tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

# Training parameters
tf.flags.DEFINE_string("schedule", "continuous_train_and_eval",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
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
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0,
                      """Fraction of GPU memory used by the process on
                      each GPU uniformly on the same machine.""")
tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                        """Allow GPU memory allocation to grow
                        dynamically.""")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        """Log the op placement to devices""")


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
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config.tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
  config.tf_config.log_device_placement = FLAGS.log_device_placement

  train_options = training_utils.TrainOptions(
      model_class=FLAGS.model,
      model_params=FLAGS.model_params)
  # On the main worker, save training options
  if config.is_chief:
    gfile.MakeDirs(output_dir)
    train_options.dump(output_dir)

  bucket_boundaries = None
  if FLAGS.buckets:
    bucket_boundaries = list(map(int, FLAGS.buckets.split(",")))

  # Training data input pipeline
  train_input_pipeline = input_pipeline.make_input_pipeline_from_def(
      def_dict=FLAGS.input_pipeline_train,
      mode=tf.contrib.learn.ModeKeys.TRAIN)

  # Create training input function
  train_input_fn = training_utils.create_input_fn(
      pipeline=train_input_pipeline,
      batch_size=FLAGS.batch_size,
      bucket_boundaries=bucket_boundaries,
      scope="train_input_fn")

  # Development data input pipeline
  dev_input_pipeline = input_pipeline.make_input_pipeline_from_def(
      def_dict=FLAGS.input_pipeline_dev,
      mode=tf.contrib.learn.ModeKeys.EVAL,
      shuffle=False, num_epochs=1)

  # Create eval input function
  eval_input_fn = training_utils.create_input_fn(
      pipeline=dev_input_pipeline,
      batch_size=FLAGS.batch_size,
      allow_smaller_final_batch=True,
      scope="dev_input_fn")


  def model_fn(features, labels, params, mode):
    """Builds the model graph"""
    model = _create_from_dict({
        "class": train_options.model_class,
        "params": train_options.model_params
    }, models, mode=mode)
    return model(features, labels, params)

  estimator = tf.contrib.learn.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=config,
      params=FLAGS.model_params)

  # Create hooks
  train_hooks = []
  for dict_ in FLAGS.hooks:
    hook = _create_from_dict(
        dict_, hooks,
        model_dir=estimator.model_dir,
        run_config=config)
    train_hooks.append(hook)

  # Create metrics
  eval_metrics = {}
  for dict_ in FLAGS.metrics:
    metric = _create_from_dict(dict_, metric_specs)
    eval_metrics[metric.name] = metric

  experiment = PatchedExperiment(
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

  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline_train = _maybe_load_yaml(FLAGS.input_pipeline_train)
  FLAGS.input_pipeline_dev = _maybe_load_yaml(FLAGS.input_pipeline_dev)

  # Load flags from config file
  final_config = {}
  if FLAGS.config_paths:
    for config_path in FLAGS.config_paths.split(","):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info("Loading config from %s", config_path)
      with gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = _deep_merge_dict(final_config, config_flags)

  tf.logging.info("Final Config:\n%s", yaml.dump(final_config))

  # Merge flags with config values
  for flag_key, flag_value in final_config.items():
    if hasattr(FLAGS, flag_key) and isinstance(getattr(FLAGS, flag_key), dict):
      merged_value = _deep_merge_dict(flag_value, getattr(FLAGS, flag_key))
      setattr(FLAGS, flag_key, merged_value)
    elif hasattr(FLAGS, flag_key):
      setattr(FLAGS, flag_key, flag_value)
    else:
      tf.logging.warning("Ignoring config flag: %s", flag_key)

  if FLAGS.save_checkpoints_secs is None \
    and FLAGS.save_checkpoints_steps is None:
    FLAGS.save_checkpoints_secs = 600
    tf.logging.info("Setting save_checkpoints_secs to %d",
                    FLAGS.save_checkpoints_secs)

  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()

  if not FLAGS.input_pipeline_train:
    raise ValueError("You must specify input_pipeline_train")

  if not FLAGS.input_pipeline_dev:
    raise ValueError("You must specify input_pipeline_dev")

  learn_runner.run(
      experiment_fn=create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
