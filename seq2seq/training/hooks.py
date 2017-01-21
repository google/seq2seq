""" Collection of SessionRunHooks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.python.training import basic_session_run_hooks, session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from seq2seq import graph_utils

class SecondOrStepTimer(basic_session_run_hooks.SecondOrStepTimer):
  """Helper class to count both seconds and steps.
  """
  pass


class MetadataCaptureHook(session_run_hook.SessionRunHook):
  """A hook to capture metadata for a single step.
  Useful for performance debugging. It performs a full trace and saves
  run_metadata and Chrome timeline information to a file.

  Args:
    output_dir: Directory to write file(s) to
    step: The step number to trace. The hook is only enable for this step.
  """

  #pylint: disable=missing-docstring

  def __init__(self, output_dir, step=10):
    self._step = step
    self._active = False
    self._global_step = None
    self.output_dir = os.path.abspath(output_dir)

  def begin(self):
    self._global_step = training_util.get_global_step()

  def before_run(self, _run_context):
    if not self._active:
      return session_run_hook.SessionRunArgs(self._global_step)
    else:
      tf.logging.info("Performing full trace on next step.")
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      return session_run_hook.SessionRunArgs(
          self._global_step, options=run_options)

  def after_run(self, _run_context, run_values):
    step_done = run_values.results
    if self._active:
      tf.logging.info("Captured full trace at step %s", step_done)
      # Create output directory
      gfile.MakeDirs(self.output_dir)

      # Save run metadata
      trace_path = os.path.join(self.output_dir, "run_meta")
      with gfile.GFile(trace_path, "wb") as trace_file:
        trace_file.write(run_values.run_metadata.SerializeToString())
        tf.logging.info("Saved run_metadata to %s", trace_path)

      # Save timeline
      timeline_path = os.path.join(self.output_dir, "timeline.json")
      with gfile.GFile(timeline_path, "w") as timeline_file:
        tl_info = timeline.Timeline(run_values.run_metadata.step_stats)
        tl_chrome = tl_info.generate_chrome_trace_format(show_memory=True)
        timeline_file.write(tl_chrome)
        tf.logging.info("Saved timeline to %s", timeline_path)

      # Save tfprof op log
      tf.contrib.tfprof.tfprof_logger.write_op_log(
          graph=tf.get_default_graph(),
          log_dir=self.output_dir,
          run_meta=run_values.run_metadata)
      tf.logging.info("Saved op log to %s", self.output_dir)
      self._active = False

    self._active = (step_done == self._step)


class TokensPerSecondCounter(session_run_hook.SessionRunHook):
  """A hooks that counts tokens/sec, where the number of tokens is
    defines as `len(source) + len(target)`.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    self._summary_tag = "tokens/sec"
    self._timer = SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    if summary_writer is None and output_dir:
      self._summary_writer = SummaryWriterCache.get(output_dir)

    self._tokens_last_step = 0


  def begin(self):
    #pylint: disable=W0201
    features = graph_utils.get_dict_from_collection("features")
    labels = graph_utils.get_dict_from_collection("labels")
    num_source_tokens = tf.reduce_sum(features["source_len"])
    num_target_tokens = tf.reduce_sum(labels["target_len"])

    self._tokens_last_step = 0
    self._global_step_tensor = training_util.get_global_step()
    self._num_tokens_tensor = num_source_tokens + num_target_tokens

    # Create a variable that stores how many tokens have been processed
    # Should be global for distributed training
    with tf.variable_scope("tokens_counter"):
      self._tokens_processed_var = tf.get_variable(
          name="count",
          shape=[],
          dtype=tf.int32,
          initializer=tf.constant_initializer(0, dtype=tf.int32))
      self._tokens_processed_add = tf.assign_add(
          self._tokens_processed_var, self._num_tokens_tensor)

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        [self._global_step_tensor, self._tokens_processed_add])

  def after_run(self, _run_context, run_values):
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


class TrainSampleHook(session_run_hook.SessionRunHook):
  """Occasionally samples predictions from the training run and prints them.

  Args:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
    sample_dir: Optional, a directory to write samples to.
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  #pylint: disable=missing-docstring

  def __init__(self, every_n_secs=None, every_n_steps=None, sample_dir=None,
               delimiter=" "):
    super(TrainSampleHook, self).__init__()
    self._sample_dir = sample_dir
    self._timer = SecondOrStepTimer(
        every_secs=every_n_secs, every_steps=every_n_steps)
    self._pred_dict = {}
    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None
    self.delimiter = delimiter

  def begin(self):
    self._iter_count = 0
    self._global_step = training_util.get_global_step()
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
      return session_run_hook.SessionRunArgs([fetches, self._global_step])
    return session_run_hook.SessionRunArgs([{}, self._global_step])

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
      result_str += self.delimiter.encode("utf-8").join(
          predicted_slice).decode("utf-8") + "\n"
      result_str += self.delimiter.encode("utf-8").join(
          target_slice).decode("utf-8") + "\n\n"
    result_str += ("=" * 100) + "\n\n"
    tf.logging.info(result_str)
    if self._sample_dir:
      filepath = os.path.join(
          self._sample_dir, "samples_{:06d}.txt".format(step))
      with gfile.GFile(filepath, "w") as file:
        file.write(result_str)
    self._timer.update_last_triggered_step(self._iter_count - 1)


class PrintModelAnalysisHook(session_run_hook.SessionRunHook):
  """Writes the parameters of the model to a file and stdout.

  Args:
    filename: The file path to write the model analysis to.
  """

  #pylint: disable=missing-docstring
  def __init__(self, filename=None):
    self.filename = filename

  def begin(self):
    # Dump to file
    opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.abspath(self.filename)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=opts)

    # Print the model analysis
    with gfile.GFile(self.filename, "r") as file:
      tf.logging.info(file.read())
