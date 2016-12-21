"""A decoder that uses beam search. Can only be used for inference, not
training.
"""

from collections import namedtuple

import tensorflow as tf
from seq2seq.inference import beam_search
from seq2seq.decoders import basic_decoder, decoder_base, attention_decoder, attention


class BeamDecoderOutput(
    namedtuple("DecoderOutput",
               ["logits", "predictions", "log_probs",
                "scores", "beam_parent_ids"])):
  """Structure for the output of a beam search decoder. This class is used
  to define the output at each step as well as the final output of the decoder.
  If used as the final output, a time dimension `T` is inserted after the
  beam_size dimension.

  Args:
    logits: Logits at the current time step of shape `[beam_size, vocab_size]`
    predictions: Chosen softmax predictions at the current time step.
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

class BeamSearchDecoder(decoder_base.DecoderBase):
  """The BeamSearchDecoder wraps another decoder to perform beam search instead
  of greedy selection. This decoder must be used with batch size of 1, which
  will result in an effective batch size of `beam_width`.

  Args:
    decoder: A instance of `DecoderBase` to be used with beam search.
    config: A `BeamSearchConfig` that defines beam search decoding parameters.
  """

  def __init__(self, decoder, config):
    super(BeamSearchDecoder, self).__init__(
        decoder.cell, decoder.max_decode_length, decoder.name)
    self.decoder = decoder
    self.config = config

  def __call__(self, *args, **kwargs):
    with self.decoder.variable_scope():
      return self._build(*args, **kwargs)

  @staticmethod
  def _wrap_loop_state(value, loop_state):
    if loop_state is None:
      return value
    else:
      return (value, loop_state)

  @staticmethod
  def _unwrap_loop_state(loop_state):
    if isinstance(loop_state, tuple) and isinstance(loop_state[0], beam_search.BeamState):
      return loop_state
    else:
      return loop_state, None

  @staticmethod
  def _pack_outputs(outputs_ta, _final_loop_state):
    """Transposes outputs from time-major to batch-major.
    """
    logits = tf.transpose(
        outputs_ta.logits.pack(), [1, 0, 2],
        name="logits")
    predictions = tf.transpose(
        outputs_ta.predictions.pack(), [1, 0],
        name="predictions")
    log_probs = tf.transpose(
        outputs_ta.log_probs.pack(), [1, 0],
        name="log_probs")
    scores = tf.transpose(
        outputs_ta.scores.pack(), [1, 0],
        name="scores")
    beam_parent_ids = tf.transpose(
        outputs_ta.beam_parent_ids.pack(), [1, 0],
        name="beam_parent_ids")
    return BeamDecoderOutput(
        logits=tf.expand_dims(logits, 0),
        predictions=tf.expand_dims(predictions, 0),
        log_probs=tf.expand_dims(log_probs, 0),
        scores=tf.expand_dims(scores, 0),
        beam_parent_ids=tf.expand_dims(beam_parent_ids, 0))

  def _step(self, time_, cell_output, cell_state, loop_state, next_input_fn):
    initial_call = (cell_output is None)

    if initial_call:
      cell_output = tf.zeros([self.config.beam_width, self.cell.output_size])
      # We start out with all beams being equal, so we tile the cell state
      # [beam_width] times
      cell_state = beam_search.nest_map(
          cell_state,
          lambda x: tf.tile(x, [self.config.beam_width, 1]))

      # Call the original decoder
      original_output = self.decoder._step(
          time_, None, cell_state, loop_state, next_input_fn)

      # Create an initial Beam State
      beam_state = beam_search.BeamState(
          time=tf.constant(0, dtype=tf.int32),
          log_probs=tf.zeros([self.config.beam_width]),
          scores=tf.zeros([self.config.beam_width]),
          predictions=tf.ones(
              [self.config.beam_width, self.decoder.max_decode_length],
              dtype=tf.int32) * -1,
          beam_parent_ids=tf.zeros([self.config.beam_width], dtype=tf.int32))

      next_loop_state = self._wrap_loop_state(
          beam_state, original_output.next_loop_state)

      # The first time we tile the initial input beam_width time
      next_input = next_input_fn(time_, None, cell_state, loop_state,
                                 original_output.outputs)
      next_input = tf.tile(next_input, [self.config.beam_width, 1])

      outputs = BeamDecoderOutput(
          logits=original_output.outputs.logits,
          predictions=original_output.outputs.predictions,
          log_probs=tf.zeros([], dtype=tf.float32),
          scores=tf.zeros([], dtype=tf.float32),
          beam_parent_ids=tf.zeros([], dtype=tf.int32))

      if "attention_context" in original_output.outputs._fields:
        next_input = tf.concat(1, [next_input,
          tf.tile(
            tf.expand_dims(original_output.outputs.attention_context, 0),
            [self.config.beam_width, 1])])

      # The final step output
      step_output = decoder_base.DecoderStepOutput(
          outputs=outputs,
          next_input=next_input,
          next_cell_state=cell_state,
          next_loop_state=next_loop_state)

    else:
      prev_beam_state, original_loop_state = self._unwrap_loop_state(loop_state)

      # Call the original decoder
      original_output = self.decoder._step(
          time_, cell_output, cell_state, original_loop_state, next_input_fn)

      # Perform a step of beam search
      beam_state = beam_search.beam_search_step(
          logits=original_output.outputs.logits,
          beam_state=prev_beam_state,
          config=self.config)
      beam_state.predictions.set_shape([None, self.decoder.max_decode_length])
      next_loop_state = self._wrap_loop_state(
          beam_state, original_output.next_loop_state)

      outputs = BeamDecoderOutput(
          logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
          predictions=tf.to_int64(beam_state.predictions[:, time_ - 1]),
          log_probs=beam_state.log_probs,
          scores=beam_state.scores,
          beam_parent_ids=beam_state.beam_parent_ids)

      # Cell states are shuffled around by beam search
      next_cell_state = beam_search.nest_map(
          original_output.next_cell_state,
          lambda x: tf.gather(x, beam_state.beam_parent_ids))

      next_input = next_input_fn(time_,
                                 (None if initial_call else cell_output),
                                 cell_state, loop_state, outputs)

      if "attention_context" in original_output.outputs._fields:
        next_input = tf.concat(1, [next_input,
                                   original_output.outputs.attention_context])

      # The final step output
      step_output = decoder_base.DecoderStepOutput(
          outputs=outputs,
          next_input=next_input,
          next_cell_state=next_cell_state,
          next_loop_state=next_loop_state)

    return step_output
