""" Collection of SessionRunHooks
"""

import os
import tensorflow as tf

from tensorflow.python.training import basic_session_run_hooks, session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile


class SecondOrStepTimer(basic_session_run_hooks._SecondOrStepTimer):
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


class TrainSampleHook(session_run_hook.SessionRunHook):
  """Occasionally samples predictions from the training run and prints them.

  Args:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
  """

  #pylint: disable=missing-docstring

  def __init__(self, every_n_secs=None, every_n_steps=None):
    super(TrainSampleHook, self).__init__()
    self._timer = SecondOrStepTimer(
        every_secs=every_n_secs, every_steps=every_n_steps)
    self.predictions_dict = {}
    self.features_dict = {}
    self.labels_dict = {}
    self.target_id_to_vocab = None
    self.predicted_words = None
    self._should_trigger = False
    self._iter_count = 0

  def begin(self):
    self._iter_count = 0
    # TODO: Is there a nicer way?
    # See https://github.com/dennybritz/seq2seq/issues/21
    self.predictions_dict = dict(
        zip(
            tf.get_collection("model_output_keys"),
            tf.get_collection("model_output_values")))
    self.features_dict = dict(
        zip(
            tf.get_collection("features_keys"),
            tf.get_collection("features_values")))
    self.labels_dict = dict(
        zip(
            tf.get_collection("labels_keys"), tf.get_collection(
                "labels_values")))
    self.target_id_to_vocab = tf.get_collection("target_id_to_vocab")[0]
    self.predicted_words = self.target_id_to_vocab.lookup(self.predictions_dict[
        "predictions"])

  def before_run(self, _run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      fetches = {
          "predicted_words": self.predicted_words,
          "target_words": self.labels_dict["target_tokens"],
          "target_len": self.labels_dict["target_len"]
      }
      return session_run_hook.SessionRunArgs(fetches)
    return None

  def after_run(self, _run_context, run_values):
    self._iter_count += 1

    if not self._should_trigger:
      return None

    # Convert dict of lists to list of dicts
    result_dict = run_values.results
    result_dicts = [
        dict(zip(result_dict, t)) for t in zip(*result_dict.values())
    ]

    # Print results
    tf.logging.info("Sampling Predictions (Prediction followed by Target)")
    tf.logging.info("=" * 100)
    for result in result_dicts:
      target_len = result["target_len"]
      predicted_slice = result["predicted_words"][:target_len]
      target_slice = result["target_words"][1:target_len]
      tf.logging.info(b" ".join(predicted_slice).decode("utf-8"))
      tf.logging.info(b" ".join(target_slice).decode("utf-8"))
      tf.logging.info("")
    self._timer.update_last_triggered_step(self._iter_count)


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
