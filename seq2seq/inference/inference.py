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

def create_predictions_iter(predictions_dict, sess):
  """Runs prediciton fetches in a sessions and flattens batches as needed to
  return an iterator of predictions. Yield elements until an
  OutOfRangeError for the feeder queues occurs.

  Args:
    predictions_dict: The dictionary to be fetched. This will be passed
      to `session.run`. The first dimensions of each element in this
      dictionary is assumed to be the batch size.
    sess: The Session to use.

  Returns:
    An iterator of the same shape as predictions_dict, but with one
    element at a time and the batch dimension removed.
  """
  with tf.contrib.slim.queues.QueueRunners(sess):
    while True:
      try:
        predictions_ = sess.run(predictions_dict)
        batch_length = list(predictions_.values())[0].shape[0]
        for i in range(batch_length):
          yield {key: value[i] for key, value in predictions_.items()}
      except tf.errors.OutOfRangeError:
        break

def create_inference_graph(
    task,
    input_pipeline,
    batch_size=32):
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

  model = task.create_model()

  # TODO: This doesn't really belong here.
  # How to get rid of this?
  if model.params["inference.beam_search.beam_width"] > 1:
    tf.logging.info("Setting batch size to 1 for beam search.")
    batch_size = 1

  input_fn = training_utils.create_input_fn(
      pipeline=input_pipeline,
      batch_size=batch_size,
      allow_smaller_final_batch=True)

  # Build the graph
  features, labels = input_fn()
  return model(
      features=features,
      labels=labels,
      params=None)
