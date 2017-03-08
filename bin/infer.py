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

""" Generates model predictions.
"""


from pydoc import locate

import yaml
from six import string_types

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq import tasks
from seq2seq.configurable import _maybe_load_yaml
from seq2seq.data import input_pipeline
from seq2seq.inference import create_inference_graph, create_predictions_iter
from seq2seq.training import utils as training_utils

tf.flags.DEFINE_string("task", "TextToTextInfer",
                       """The type of inference task to run. Must be defined
                       in seq2seq.tasks""")
tf.flags.DEFINE_string("task_params", "{}",
                       """Parameters to pass to the task class.
                       A YAML/JSON string.""")

tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

tf.flags.DEFINE_string("input_pipeline", None,
                       """Defines how input data should be loaded.
                       A YAML string.""")

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")

FLAGS = tf.flags.FLAGS

def main(_argv):
  """Program entrypoint.
  """

  # Load flags from config file
  if FLAGS.config_path:
    with gfile.GFile(FLAGS.config_path) as config_file:
      config_flags = yaml.load(config_file)
      for flag_key, flag_value in config_flags.items():
        setattr(FLAGS, flag_key, flag_value)

  if isinstance(FLAGS.input_pipeline, string_types):
    FLAGS.input_pipeline = _maybe_load_yaml(FLAGS.input_pipeline)

  input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
      FLAGS.input_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
      shuffle=False, num_epochs=1)

  # Load saved training options
  train_options = training_utils.TrainOptions.load(FLAGS.model_dir)

  # Load the inference task class
  task_class = locate(FLAGS.task) or getattr(tasks, FLAGS.task)
  task = task_class(
      params=_maybe_load_yaml(FLAGS.task_params),
      train_options=train_options)

  # Create the graph used for inference
  predictions, _, _ = create_inference_graph(
      task=task,
      input_pipeline=input_pipeline_infer,
      batch_size=FLAGS.batch_size)

  # Define which tensors to fetch
  prediction_keys = task.prediction_keys()
  predictions = {k: v for k, v in predictions.items() if k in prediction_keys}

  saver = tf.train.Saver()

  checkpoint_path = FLAGS.checkpoint_path
  if not checkpoint_path:
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    # Restore checkpoint
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Restored model from %s", checkpoint_path)

    # Process predictions
    predictions_iter = create_predictions_iter(predictions, sess)
    task.begin()
    for idx, predictions_dict in enumerate(predictions_iter):
      task.process_batch(idx, predictions_dict)
    task.end()

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
