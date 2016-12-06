""" Collection of SessionRunHooks
"""

import os
import itertools
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn import basic_session_run_hooks, session_run_hook

class SecondOrStepTimer(basic_session_run_hooks.basic_session_run_hooks._SecondOrStepTimer):
  pass

class TrainSampleHook(session_run_hook.SessionRunHook):
  def __init__(self, every_n_secs=None, every_n_steps=None):
    self._timer = SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_steps)
    self.predictions_dict = {}
    self.features_dict = {}
    self.labels_dict = {}
    self.target_id_to_vocab = None
    self.predicted_words = None
    self._should_trigger = False
    self._iter_count = 0

  def begin(self):
    self._iter_count = 0
    self.predictions_dict = dict(zip(
      tf.get_collection("model_output_keys"),
      tf.get_collection("model_output_values")))
    self.features_dict = dict(zip(
      tf.get_collection("features_keys"),
      tf.get_collection("features_values")))
    self.labels_dict = dict(zip(
      tf.get_collection("labels_keys"),
      tf.get_collection("labels_values")))
    self.target_id_to_vocab = tf.get_collection("target_id_to_vocab")[0]
    self.predicted_words = self.target_id_to_vocab.lookup(self.predictions_dict["predictions"])

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
    result_dicts = [dict(zip(result_dict, t)) for t in zip(*result_dict.values())]

    tf.logging.info("Sampling Predictions (Prediction followed by Target)")
    tf.logging.info("=" * 100)
    for result in result_dicts:
      target_len = result["target_len"]
      predicted_slice = result["predicted_words"][:target_len]
      target_slice = result["target_words"][1:target_len]
      tf.logging.info(" ".join(predicted_slice.astype(np.str)))
      tf.logging.info(" ".join(target_slice.astype(np.str)))
      tf.logging.info("")
    self._timer.update_last_triggered_step(self._iter_count)



class PrintModelAnalysisHook(session_run_hook.SessionRunHook):
  """A SessionRunHook that writes the parameters of the model to a file and stdout.

  Args:
    filename: The file path to write the model analysis to.
  """

  def __init__(self, filename=None):
    self.filename = filename

  def begin(self):
    # Dump to file
    opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.abspath(self.filename)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(), tfprof_options=opts)

    # Print the model analysis
    with open(self.filename, "r") as file:
      tf.logging.info(file.read())
