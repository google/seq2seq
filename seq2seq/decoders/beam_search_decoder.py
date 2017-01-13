"""A decoder that uses beam search. Can only be used for inference, not
training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
from seq2seq.inference import beam_search
from seq2seq.decoders.decoder_base import DecoderBase, DecoderStepOutput


class BeamDecoderOutput(
    namedtuple("DecoderOutput", [
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


class BeamSearchDecoder(DecoderBase):
  """The BeamSearchDecoder wraps another decoder to perform beam search instead
  of greedy selection. This decoder must be used with batch size of 1, which
  will result in an effective batch size of `beam_width`.

  Args:
    decoder: A instance of `DecoderBase` to be used with beam search.
    config: A `BeamSearchConfig` that defines beam search decoding parameters.
  """

  def __init__(self, decoder, config):
    super(BeamSearchDecoder, self).__init__(
        decoder.cell, decoder.input_fn, decoder.max_decode_length,
        decoder.prediction_fn, decoder.name)
    self.decoder = decoder
    self.config = config

  def __call__(self, *args, **kwargs):
    with self.decoder.variable_scope():
      return self._build(*args, **kwargs)

  @staticmethod
  def _wrap_loop_state(value, loop_state):
    """Wraps the loop state to also include to beam decoder state.
    """
    if loop_state is None:
      return value
    else:
      return (value, loop_state)

  @staticmethod
  def _unwrap_loop_state(loop_state):
    """Unwraps the loop state to return (beam_decoder_state, original_state)
    """
    if isinstance(loop_state, tuple) and isinstance(loop_state[0],
                                                    beam_search.BeamState):
      return loop_state
    else:
      return loop_state, None

  def pack_outputs(self, outputs_ta, final_loop_state):
    """Transposes outputs from time-major to batch-major.
    """
    logits = outputs_ta.logits.stack()
    predicted_ids = outputs_ta.predicted_ids.stack()
    log_probs = outputs_ta.log_probs.stack()
    scores = outputs_ta.scores.stack()
    beam_parent_ids = outputs_ta.beam_parent_ids.stack()

    _, original_final_loop_state = self._unwrap_loop_state(final_loop_state)
    orignal_output = self.decoder.pack_outputs(
        outputs_ta.original_outputs, original_final_loop_state)

    # We're using a batch size of 1, so we add an extra dimension to
    # convert tensors to [1, beam_width, ...] shape. This way Tensorflow
    # doesn't confuse batch_size with beam_width
    orignal_output = orignal_output.__class__(
        *[tf.expand_dims(_, 1) for _ in orignal_output]
    )

    return BeamDecoderOutput(
        logits=tf.expand_dims(logits, 1),
        predicted_ids=tf.expand_dims(predicted_ids, 1),
        log_probs=tf.expand_dims(log_probs, 1),
        scores=tf.expand_dims(scores, 1),
        beam_parent_ids=tf.expand_dims(beam_parent_ids, 1),
        original_outputs=orignal_output)

  def compute_output(self, cell_output):
    raise ValueError("""Beam Search decoder does not support this method.""")

  def output_shapes(self):
    return BeamDecoderOutput(
        logits=tf.zeros([self.decoder.vocab_size]),
        predicted_ids=tf.zeros([], dtype=tf.int64),
        log_probs=tf.zeros([], dtype=tf.float32),
        scores=tf.zeros([], dtype=tf.float32),
        beam_parent_ids=tf.zeros([], dtype=tf.int32),
        original_outputs=self.decoder.output_shapes())

  def create_next_input(self, time_, initial_call, output):
    if initial_call:
      next_input, elements_finished = self.decoder.create_next_input(
          time_, initial_call, output.original_outputs)
      # The first time we tile the initial input [beam_width] time
      next_input_beam = tf.tile(next_input, [self.config.beam_width, 1])
      elements_finished_beam = tf.tile(
          elements_finished, [self.config.beam_width])
      return next_input_beam, elements_finished_beam

    # Shuffle the original output according to our beam search result
    original_values_shuffled = []
    for value in output.original_outputs:
      value_shuffled = beam_search.nest_map(
          value,
          lambda x: tf.gather(x, output.beam_parent_ids))
      original_values_shuffled.append(value_shuffled)
    original_outputs_shuffled = output.original_outputs.__class__(
        *original_values_shuffled)
    original_outputs_shuffled = original_outputs_shuffled._replace(
        predicted_ids=output.predicted_ids)

    next_input, elements_finished = self.decoder.create_next_input(
        time_, initial_call, original_outputs_shuffled)
    return next_input, elements_finished

  def step(self, time_, cell_output, cell_state, loop_state):
    initial_call = (cell_output is None)

    if initial_call:
      cell_output = tf.zeros([self.config.beam_width, self.cell.output_size])

      # We start out with all beams being equal, so we tile the cell state
      # [beam_width] times
      next_cell_state = beam_search.nest_map(
          cell_state, lambda x: tf.tile(x, [self.config.beam_width, 1]))

      # Call the original decoder
      original_outputs = self.decoder.step(
          time_, None, cell_state, loop_state)

      # Create an initial Beam State
      beam_state = beam_search.create_initial_beam_state(
          config=self.config,
          max_time=self.decoder.max_decode_length)

      next_loop_state = self._wrap_loop_state(
          beam_state, original_outputs.next_loop_state)

      outputs = self.output_shapes()

    else:
      prev_beam_state, original_loop_state = self._unwrap_loop_state(loop_state)

      # Call the original decoder
      original_outputs = self.decoder.step(
          time_, cell_output, cell_state, original_loop_state)

      # Perform a step of beam search
      beam_state = beam_search.beam_search_step(
          logits=original_outputs.outputs.logits,
          beam_state=prev_beam_state,
          config=self.config)
      beam_state.predicted_ids.set_shape([None, self.decoder.max_decode_length])
      next_loop_state = self._wrap_loop_state(
          beam_state, original_outputs.next_loop_state)

      outputs = BeamDecoderOutput(
          logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
          predicted_ids=tf.to_int64(beam_state.predicted_ids[:, time_ - 1]),
          log_probs=beam_state.log_probs,
          scores=beam_state.scores,
          beam_parent_ids=beam_state.beam_parent_ids,
          original_outputs=original_outputs.outputs)

      # Cell states are shuffled around by beam search
      next_cell_state = beam_search.nest_map(
          original_outputs.next_cell_state,
          lambda x: tf.gather(x, beam_state.beam_parent_ids))

    # The final step output
    step_output = DecoderStepOutput(
        outputs=outputs,
        next_cell_state=next_cell_state,
        next_loop_state=next_loop_state)

    return step_output
