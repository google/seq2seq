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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq.training import utils as training_utils


def create_inference_graph(model, input_pipeline, batch_size=32):
  """Creates a graph to perform inference.

  Args:
    task: An `InferenceTask` instance.
    input_pipeline: An instance of `InputPipeline` that defines
      how to read and parse data.
    batch_size: The batch size used for inference

  Returns:
    The return value of the model function, typically a tuple of
    (predictions, loss, train_op).
  """

  # TODO: This doesn't really belong here.
  # How to get rid of this?
  if hasattr(model, "use_beam_search"):
    if model.use_beam_search:
      tf.logging.info("Setting batch size to 1 for beam search.")
      batch_size = 1

  input_fn = training_utils.create_input_fn(
      pipeline=input_pipeline,
      batch_size=batch_size,
      allow_smaller_final_batch=True)

  # Build the graph
  features, labels = input_fn()
  return model(features=features, labels=labels, params=None)
