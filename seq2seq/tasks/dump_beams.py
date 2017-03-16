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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict


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
    params.update({"file": "",})
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
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      self._beam_accum["predicted_ids"].append(fetches[
          "beam_search_output.predicted_ids"])
      self._beam_accum["beam_parent_ids"].append(fetches[
          "beam_search_output.beam_parent_ids"])
      self._beam_accum["scores"].append(fetches["beam_search_output.scores"])
      self._beam_accum["log_probs"].append(fetches[
          "beam_search_output.log_probs"])

  def end(self, _session):
    np.savez(self.params["file"], **self._beam_accum)
