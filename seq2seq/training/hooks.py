# -*- coding: utf-8 -*-
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
""" Collection of tf.train.SessionRunHooks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os

import six
import yaml

import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer  # pylint: disable=E0611
from tensorflow.python.training.summary_io import SummaryWriterCache  # pylint: disable=E0611
from tensorflow.python.client import timeline  # pylint: disable=E0611
from tensorflow import gfile

from seq2seq.configurable import Configurable, abstractstaticmethod
from seq2seq import graph_utils


@six.add_metaclass(abc.ABCMeta)
class TrainingHook(tf.train.SessionRunHook, Configurable):
  """Abstract base class for training hooks.
  """

  def __init__(self, params, model_dir, is_chief=True):
    tf.train.SessionRunHook.__init__(self)
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.TRAIN)
    self._model_dir = model_dir
    self._is_chief = is_chief

  @property
  def model_dir(self):
    """Returns the directory model checkpoints are written to.
    """
    return os.path.abspath(self._model_dir)

  @property
  def is_chief(self):
    """Returns true if and only if the current process is the chief.
    This is used for distributed training.
    """
    return self._is_chief

  @abstractstaticmethod
  def default_params():
    raise NotImplementedError()


class MetadataCaptureHook(TrainingHook):
  """A hook to capture metadata for a single step.
  Useful for performance debugging. It performs a full trace and saves
  run_metadata and Chrome timeline information to a file.

  Args:
    step: The step number to trace. The hook is only enable for this step.
  """

  def __init__(self, params, model_dir, is_chief=True):
    super(MetadataCaptureHook, self).__init__(params, model_dir, is_chief)
    self._active = False
    self._done = False
    self._global_step = None
    self._output_dir = os.path.abspath(self.model_dir)

  @staticmethod
  def default_params():
    return {"step": 10}

  def begin(self):
    self._global_step = tf.train.get_global_step()

  def before_run(self, _run_context):
    if not self.is_chief or self._done:
      return
    if not self._active:
      return tf.train.SessionRunArgs(self._global_step)
    else:
      tf.logging.info("Performing full trace on next step.")
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      return tf.train.SessionRunArgs(self._global_step, options=run_options)

  def after_run(self, _run_context, run_values):
    if not self.is_chief or self._done:
      return

    step_done = run_values.results
    if self._active:
      tf.logging.info("Captured full trace at step %s", step_done)
      # Create output directory
      gfile.MakeDirs(self._output_dir)

      # Save run metadata
      trace_path = os.path.join(self._output_dir, "run_meta")
      with gfile.GFile(trace_path, "wb") as trace_file:
        trace_file.write(run_values.run_metadata.SerializeToString())
        tf.logging.info("Saved run_metadata to %s", trace_path)

      # Save timeline
      timeline_path = os.path.join(self._output_dir, "timeline.json")
      with gfile.GFile(timeline_path, "w") as timeline_file:
        tl_info = timeline.Timeline(run_values.run_metadata.step_stats)
        tl_chrome = tl_info.generate_chrome_trace_format(show_memory=True)
        timeline_file.write(tl_chrome)
        tf.logging.info("Saved timeline to %s", timeline_path)

      # Save tfprof op log
      tf.contrib.tfprof.tfprof_logger.write_op_log(
          graph=tf.get_default_graph(),
          log_dir=self._output_dir,
          run_meta=run_values.run_metadata)
      tf.logging.info("Saved op log to %s", self._output_dir)
      self._active = False
      self._done = True

    self._active = (step_done >= self.params["step"])


class TokensPerSecondCounter(TrainingHook):
  """A hooks that counts tokens/sec, where the number of tokens is
    defines as `len(source) + len(target)`.
  """

  def __init__(self, params, model_dir, is_chief=True, summary_writer=None):
    super(TokensPerSecondCounter, self).__init__(params, model_dir, is_chief)

    self._summary_tag = "tokens/sec"
    self._timer = SecondOrStepTimer(
        every_steps=self.params["every_n_steps"],
        every_secs=self.params["every_n_secs"])

    self._summary_writer = summary_writer
    if summary_writer is None and self.model_dir:
      self._summary_writer = SummaryWriterCache.get(self.model_dir)

    self._tokens_last_step = 0

  @staticmethod
  def default_params():
    return {"every_n_steps": 100, "every_n_secs": None}

  def begin(self):
    if not self.is_chief:
      return

    #pylint: disable=W0201
    features = graph_utils.get_dict_from_collection("features")
    labels = graph_utils.get_dict_from_collection("labels")

    self._num_tokens_tensor = tf.constant(0)
    if "source_len" in features:
      self._num_tokens_tensor += tf.reduce_sum(features["source_len"])
    if "target_len" in labels:
      self._num_tokens_tensor += tf.reduce_sum(labels["target_len"])

    self._tokens_last_step = 0
    self._global_step_tensor = tf.train.get_global_step()

    # Create a variable that stores how many tokens have been processed
    # Should be global for distributed training
    with tf.variable_scope("tokens_counter"):
      self._tokens_processed_var = tf.get_variable(
          name="count",
          shape=[],
          dtype=tf.int32,
          initializer=tf.constant_initializer(
              0, dtype=tf.int32))
      self._tokens_processed_add = tf.assign_add(self._tokens_processed_var,
                                                 self._num_tokens_tensor)

  def before_run(self, run_context):
    if not self.is_chief:
      return

    return tf.train.SessionRunArgs(
        [self._global_step_tensor, self._tokens_processed_add])

  def after_run(self, _run_context, run_values):
    if not self.is_chief:
      return

    global_step, num_tokens = run_values.results
    tokens_processed = num_tokens - self._tokens_last_step

    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, _ = self._timer.update_last_triggered_step(global_step)
      if elapsed_time is not None:
        tokens_per_sec = tokens_processed / elapsed_time
        if self._summary_writer is not None:
          summary = tf.Summary(value=[
              tf.Summary.Value(
                  tag=self._summary_tag, simple_value=tokens_per_sec)
          ])
          self._summary_writer.add_summary(summary, global_step)
        tf.logging.info("%s: %g", self._summary_tag, tokens_per_sec)
      self._tokens_last_step = num_tokens


class TrainSampleHook(TrainingHook):
  """Occasionally samples predictions from the training run and prints them.

  Params:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
    sample_dir: Optional, a directory to write samples to.
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  #pylint: disable=missing-docstring

  def __init__(self, params, model_dir, is_chief=True):
    super(TrainSampleHook, self).__init__(params, model_dir, is_chief)
    self._sample_dir = os.path.join(self.model_dir, "samples")
    self._timer = SecondOrStepTimer(
        every_secs=self.params["every_n_secs"],
        every_steps=self.params["every_n_steps"])
    self._pred_dict = {}
    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None
    self._source_delimiter = self.params["source_delimiter"]
    self._target_delimiter = self.params["target_delimiter"]

  @staticmethod
  def default_params():
    return {
        "every_n_secs": None,
        "every_n_steps": 1000,
        "source_delimiter": " ",
        "target_delimiter": " "
    }

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    self._pred_dict = graph_utils.get_dict_from_collection("predictions")
    # Create the sample directory
    if self._sample_dir is not None:
      gfile.MakeDirs(self._sample_dir)

  def before_run(self, _run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      fetches = {
          "predicted_tokens": self._pred_dict["predicted_tokens"],
          "target_words": self._pred_dict["labels.target_tokens"],
          "target_len": self._pred_dict["labels.target_len"]
      }
      return tf.train.SessionRunArgs([fetches, self._global_step])
    return tf.train.SessionRunArgs([{}, self._global_step])

  def after_run(self, _run_context, run_values):
    result_dict, step = run_values.results
    self._iter_count = step

    if not self._should_trigger:
      return None

    # Convert dict of lists to list of dicts
    result_dicts = [
        dict(zip(result_dict, t)) for t in zip(*result_dict.values())
    ]

    # Print results
    result_str = ""
    result_str += "Prediction followed by Target @ Step {}\n".format(step)
    result_str += ("=" * 100) + "\n"
    for result in result_dicts:
      target_len = result["target_len"]
      predicted_slice = result["predicted_tokens"][:target_len - 1]
      target_slice = result["target_words"][1:target_len]
      result_str += self._target_delimiter.encode("utf-8").join(
          predicted_slice).decode("utf-8") + "\n"
      result_str += self._target_delimiter.encode("utf-8").join(
          target_slice).decode("utf-8") + "\n\n"
    result_str += ("=" * 100) + "\n\n"
    tf.logging.info(result_str)
    if self._sample_dir:
      filepath = os.path.join(self._sample_dir,
                              "samples_{:06d}.txt".format(step))
      with gfile.GFile(filepath, "w") as file:
        file.write(result_str)
    self._timer.update_last_triggered_step(self._iter_count - 1)


class PrintModelAnalysisHook(TrainingHook):
  """Writes the parameters of the model to a file and stdout.
  """

  #pylint: disable=missing-docstring
  def __init__(self, params, model_dir, is_chief=True):
    super(PrintModelAnalysisHook, self).__init__(params, model_dir, is_chief)
    self._filename = os.path.join(self.model_dir, "model_analysis.txt")

  @staticmethod
  def default_params():
    return {}

  def begin(self):
    # Dump to file on the chief worker
    if self.is_chief:
      opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
      opts['dump_to_file'] = os.path.abspath(self._filename)
      tf.contrib.tfprof.model_analyzer.print_model_analysis(
          tf.get_default_graph(), tfprof_options=opts)

    # Print the model analysis
    with gfile.GFile(self._filename) as file:
      tf.logging.info(file.read())


class VariableRestoreHook(TrainingHook):
  """A hooks that restored variables from a given checkpoints.

  Params:
    prefix: Variables matching this prefix are restored.
    checkpoint_path: Path to the checkpoint to restore variables from.
  """

  def __init__(self, params, model_dir, is_chief=True):
    super(VariableRestoreHook, self).__init__(params, model_dir, is_chief)
    self._saver = None

  @staticmethod
  def default_params():
    return {"prefix": "", "checkpoint_path": ""}

  def begin(self):
    variables = tf.contrib.framework.get_variables(scope=self.params["prefix"])

    def varname_in_checkpoint(name):
      """Removes the prefix from the variable name.
      """
      prefix_parts = self.params["prefix"].split("/")
      checkpoint_prefix = "/".join(prefix_parts[:-1])
      return name.replace(checkpoint_prefix + "/", "")

    target_names = [varname_in_checkpoint(_.op.name) for _ in variables]
    restore_map = {k: v for k, v in zip(target_names, variables)}

    tf.logging.info("Restoring variables: \n%s",
                    yaml.dump({k: v.op.name
                               for k, v in restore_map.items()}))

    self._saver = tf.train.Saver(restore_map)

  def after_create_session(self, session, coord):
    self._saver.restore(session, self.params["checkpoint_path"])
    tf.logging.info("Successfully restored all variables")
