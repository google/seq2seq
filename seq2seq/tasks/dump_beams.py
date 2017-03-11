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

"""
Task where both the input and output sequence are plain text.
"""

import functools
import os

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from seq2seq.tasks.decode_text import _get_prediction_length
from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict
from seq2seq.training import hooks


def _get_scores(predictions_dict):
  """Returns the attention scores, sliced by source and target length.
  """
  prediction_len = _get_prediction_length(predictions_dict)
  source_len = predictions_dict["features.source_len"]
  return predictions_dict["attention_scores"][:prediction_len, :source_len]

def _create_figure(predictions_dict):
  """Creates and returns a new figure that visualizes
  attention scores for for a single model predictions.
  """

  # Find out how long the predicted sequence is
  target_words = list(predictions_dict["predicted_tokens"])

  prediction_len = _get_prediction_length(predictions_dict)

  # Get source words
  source_len = predictions_dict["features.source_len"]
  source_words = predictions_dict["features.source_tokens"][:source_len]

  # Plot
  fig = plt.figure(figsize=(8, 8))
  plt.imshow(
      X=predictions_dict["attention_scores"][:prediction_len, :source_len],
      interpolation="nearest",
      cmap=plt.cm.Blues)
  plt.xticks(np.arange(source_len), source_words, rotation=45)
  plt.yticks(np.arange(prediction_len), target_words, rotation=-45)
  fig.tight_layout()

  return fig

class DumpBeams(InferenceTask):
  """Defines inference for tasks where both the input and output sequences
  are plain text.

  Params:
    file: File to write beam search information to.
  """

  def __init__(self, params):
    super(DumpBeams, self).__init__(params)
    self._beam_accum = {
        "predicted_ids": [],
        "beam_parent_ids": [],
        "scores": [],
        "log_probs": []
    }

    if not self.params["file"]:
      raise ValueError("Must specify file for DumpBeams")

  @staticmethod
  def default_params():
    params = {}
    params.update({
        "file": "",
    })
    return params

  def before_run(self, _run_context):
    fetches = {}
    fetches["beam_search_output.predicted_ids"] = self._predictions[
        "beam_search_output.predicted_ids"]
    fetches["beam_search_output.beam_parent_ids"] = self._predictions[
        "beam_search_output.beam_parent_ids"]
    fetches["beam_search_output.scores"] = self._predictions[
        "beam_search_output.scores"]
    fetches["beam_search_output.log_probs"] = self._predictions[
        "beam_search_output.log_probs"]
    return SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      self._beam_accum["predicted_ids"].append(fetches[
          "beam_search_output.predicted_ids"])
      self._beam_accum["beam_parent_ids"].append(fetches[
          "beam_search_output.beam_parent_ids"])
      self._beam_accum["scores"].append(fetches[
          "beam_search_output.scores"])
      self._beam_accum["log_probs"].append(fetches[
          "beam_search_output.log_probs"])

  def end(self, _session):
    np.savez(self.params["file"], **self._beam_accum)
