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
"""A decoder that uses beam search. Can only be used for inference, not
training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611

from seq2seq.inference import beam_search
from seq2seq.decoders.rnn_decoder import RNNDecoder


class FinalBeamDecoderOutput(
    namedtuple("FinalBeamDecoderOutput",
               ["predicted_ids", "beam_search_output"])):
  """Final outputs returned by the beam search after all decoding is finished.

  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[T, 1, beam_width]`.
    beam_search_output: An instance of `BeamDecoderOutput` that describes
      the state of the beam search.
  """
  pass


class BeamDecoderOutput(
    namedtuple("BeamDecoderOutput", [
        "logits", "predicted_ids", "log_probs", "scores", "beam_parent_ids",
        "original_outputs"
    ])):
  """Structure for the output of a beam search decoder. This class is used
  to define the output at each step as well as the final output of the decoder.
  If used as the final output, a time dimension `T` is inserted after the
  beam_size dimension.

  Args:
    logits: Logits at the current time step of shape `[beam_size, vocab_size]`
    predicted_ids: Chosen softmax predictions at the current time step.
      An int32 tensor of shape `[beam_size]`.
    log_probs: Total log probabilities of all beams at the current time step.
      A float32 tensor of shaep `[beam_size]`.
    scores: Total scores of all beams at the current time step. This differs
      from log probabilities in that the score may add additional processing
      such as length normalization. A float32 tensor of shape `[beam_size]`.
    beam_parent_ids: The indices of the beams that are being continued.
      An int32 tensor of shape `[beam_size]`.
  """
  pass


class BeamSearchDecoder(RNNDecoder):
  """The BeamSearchDecoder wraps another decoder to perform beam search instead
  of greedy selection. This decoder must be used with batch size of 1, which
  will result in an effective batch size of `beam_width`.

  Args:
    decoder: A instance of `RNNDecoder` to be used with beam search.
    config: A `BeamSearchConfig` that defines beam search decoding parameters.
  """

  def __init__(self, decoder, config):
    super(BeamSearchDecoder, self).__init__(decoder.params, decoder.mode,
                                            decoder.name)
    self.decoder = decoder
    self.config = config

  def __call__(self, *args, **kwargs):
    with self.decoder.variable_scope():
      return self._build(*args, **kwargs)

  @property
  def output_size(self):
    return BeamDecoderOutput(
        logits=self.decoder.vocab_size,
        predicted_ids=tf.TensorShape([]),
        log_probs=tf.TensorShape([]),
        scores=tf.TensorShape([]),
        beam_parent_ids=tf.TensorShape([]),
        original_outputs=self.decoder.output_size)

  @property
  def output_dtype(self):
    return BeamDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        log_probs=tf.float32,
        scores=tf.float32,
        beam_parent_ids=tf.int32,
        original_outputs=self.decoder.output_dtype)

  @property
  def batch_size(self):
    return self.config.beam_width

  def initialize(self, name=None):
    finished, first_inputs, initial_state = self.decoder.initialize()

    # Create beam state
    beam_state = beam_search.create_initial_beam_state(config=self.config)
    return finished, first_inputs, (initial_state, beam_state)

  def finalize(self, outputs, final_state):
    # Gather according to beam search result
    predicted_ids = beam_search.gather_tree(outputs.predicted_ids,
                                            outputs.beam_parent_ids)

    # We're using a batch size of 1, so we add an extra dimension to
    # convert tensors to [1, beam_width, ...] shape. This way Tensorflow
    # doesn't confuse batch_size with beam_width
    outputs = nest.map_structure(lambda x: tf.expand_dims(x, 1), outputs)

    final_outputs = FinalBeamDecoderOutput(
        predicted_ids=tf.expand_dims(predicted_ids, 1),
        beam_search_output=outputs)

    return final_outputs, final_state

  def _build(self, initial_state, helper):
    # Tile initial state
    initial_state = nest.map_structure(
        lambda x: tf.tile(x, [self.batch_size, 1]), initial_state)
    self.decoder._setup(initial_state, helper)  #pylint: disable=W0212
    return super(BeamSearchDecoder, self)._build(self.decoder.initial_state,
                                                 self.decoder.helper)

  def step(self, time_, inputs, state, name=None):
    decoder_state, beam_state = state

    # Call the original decoder
    (decoder_output, decoder_state, _, _) = self.decoder.step(time_, inputs,
                                                              decoder_state)

    # Perform a step of beam search
    bs_output, beam_state = beam_search.beam_search_step(
        time_=time_,
        logits=decoder_output.logits,
        beam_state=beam_state,
        config=self.config)

    # Shuffle everything according to beam search result
    decoder_state = nest.map_structure(
        lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_state)
    decoder_output = nest.map_structure(
        lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_output)

    next_state = (decoder_state, beam_state)

    outputs = BeamDecoderOutput(
        logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
        predicted_ids=bs_output.predicted_ids,
        log_probs=beam_state.log_probs,
        scores=bs_output.scores,
        beam_parent_ids=bs_output.beam_parent_ids,
        original_outputs=decoder_output)

    finished, next_inputs, next_state = self.decoder.helper.next_inputs(
        time=time_,
        outputs=decoder_output,
        state=next_state,
        sample_ids=bs_output.predicted_ids)
    next_inputs.set_shape([self.batch_size, None])

    return (outputs, next_state, next_inputs, finished)
