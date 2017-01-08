"""In-Graph Beam Search Implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
from tensorflow.python.util import nest


class BeamState(
    namedtuple("BeamState", [
        "time", "log_probs", "scores", "predicted_ids", "beam_parent_ids"
    ])):
  """Dscribes the state at each step of a beam search.

  Args:
    time: The current time step of the RNN, starting from 0. An int32.
    log_probs: The *total* log probabilities of all beams at the current step.
      This the sum of the all log probabilities over previous steps.
      A float32 tensor of shape `[beam_width]`.
    scores: The *total* scores of all beams at the current step. This is
      different from the log probabilities in that the score may include
      extra computation, e.g. length normalization.
      A float32 tensor of shape `[beam_width]`.
    predicted_ids: The chosen word ids at the current time step. An int32 tensor
      of shape `[beam_width]`.
    beam_parent_ids: The indices of the continued parent beams.
      An int32 tensor of shape `[beam_width]`.
  """
  pass


class BeamSearchConfig(
    namedtuple("BeamSearchConfig", [
        "beam_width", "vocab_size", "eos_token", "score_fn",
        "choose_successors_fn"
    ])):
  """Configuration object for beam search.

  Args:
    beam_width: Number of beams to use, an integer
    vocab_size: Output vocabulary size
    eos_token: The id of the EOS token, used to mark beams as "done"
    score_fn: A function used to calculate the score for each beam.
      Should map from (log_probs, sequence_lengths) => score.
    choose_successors_fn: A function used to choose beam successors based
      on their scores. Maps from (scores, config) => (chosen scores, chosen_ids)
  """
  pass

def create_initial_beam_state(config, max_time):
  """Creates an instance of `BeamState` that can be used on the first
  call to `beam_step`.

  Args:
    config: A BeamSearchConfig
    max_time: Maximum number of beam search steps. This is used to define
      the shape of the predictions: `[beam_width, max_time]`.

  Returns:
    An instance of `BeamState`.
  """
  return BeamState(
      time=tf.constant(0, dtype=tf.int32),
      log_probs=tf.zeros([config.beam_width]),
      scores=tf.zeros([config.beam_width]),
      predicted_ids=tf.ones([config.beam_width, max_time], dtype=tf.int32) * -1,
      beam_parent_ids=tf.zeros([config.beam_width], dtype=tf.int32))

def logprob_score(log_probs, _sequence_lengths):
  """A scoring function where the beam score is equal to the log probability.
  """
  return log_probs


def choose_top_k(scores_flat, config):
  """Chooses the top-k beams as successors.
  """
  next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=config.beam_width)
  return next_beam_scores, word_indices


def nest_map(inputs, map_fn, name=None):
  """Applies a function to (possibly nested) tuple of tensors.
  """
  if nest.is_sequence(inputs):
    inputs_flat = nest.flatten(inputs)
    y_flat = [map_fn(_) for _ in inputs_flat]
    outputs = nest.pack_sequence_as(inputs, y_flat)
  else:
    outputs = map_fn(inputs)
  if name:
    outputs = tf.identity(outputs, name=name)
  return outputs


def mask_probs(probs, eos_token, finished):
  """Masks log probabilities such that finished beams
  allocate all probability mass to eos. Unfinished beams remain unchanged.

  Args:
    probs: Log probabiltiies of shape `[beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to
    finished: A boolean tensor of shape `[beam_width]` that specifies which
      elements in the beam are finished already.

  Returns:
    A tensor of shape `[beam_width, vocab_size]`, where unfinished beams
    stay unchanged and finished beams are replaced with a tensor that has all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[1]
  finished_mask = tf.expand_dims(tf.to_float(1. - tf.to_float(finished)), 1)
  # These examples are not finished and we leave them
  non_finished_examples = finished_mask * probs
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=tf.float32,
      on_value=0.,
      off_value=tf.float32.min)
  finished_examples = (1. - finished_mask) * finished_row
  return finished_examples + non_finished_examples


def sequence_length(sequence, eos_token, include_eos_in_length=False):
  """Calculates the sequence length using an EOS token as the endpoint.

  Args:
    sequence: The input sequences. A tensor of shape `[B, T]`
    eos_token: An element that marks the end of a sequence. Must be of the same
      dtype as `sequence`.
    include_eos_in_length: If true, the returned length includes the EOS token.
      By default the EOS token is not included in the length.

  Returns:
    An int32 tensor of shape [B] where each element is the length of the
    corresponding input sequence. If no EOS token is found in a sequence
    its length is equal to T.
  """

  def single_sequence_length(sequence, eos_token, include_eos_in_length):
    """Calculats the length for a single sequence"""
    indices = tf.where(tf.equal(sequence, eos_token))
    return tf.cond(
        tf.size(indices) > 0,
        lambda: tf.to_int32(tf.reduce_min(indices)) + \
            tf.to_int32(include_eos_in_length),
        lambda: tf.to_int32(tf.size(sequence)))

  return tf.map_fn(
      lambda s: single_sequence_length(s, eos_token, include_eos_in_length),
      sequence)


def beam_search_step(logits, beam_state, config):
  """Performs a single step of Beam Search Decoding.

  Args:
    logits: Logits at the current time step. A tensor of shape `[B, vocab_size]`
    beam_state: Current state of the beam search. An instance of `BeamState`
    config: An instance of `BeamSearchConfig`

  Returns:
    A new beam state.
  """

  # Time starts at 1 (with all predictions having length 0)
  time_ = beam_state.time + 1

  # Calculate the current lengths of the predictions
  prediction_lengths = sequence_length(
      beam_state.predicted_ids, config.eos_token, False)
  prediction_lengths = tf.to_int32(prediction_lengths)

  # Find all beams that are "finished" (i.e. have an EOS token already)
  previously_finished = (prediction_lengths < time_ - 1)

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [beam_width, vocab_size]
  probs = tf.nn.log_softmax(logits)
  probs = mask_probs(probs, config.eos_token, previously_finished)
  total_probs = tf.expand_dims(beam_state.log_probs, 1) + probs

  # Flatten tensors. Shape: [beam_size * vocab_size]
  total_probs_flat = tf.reshape(total_probs, [-1], name="total_probs_flat")

  # Calculate the new predictions lengths
  # We add 1 to all continuations that are not EOS
  lengths_to_add = tf.tile(
      tf.expand_dims(
          tf.one_hot(
              config.eos_token, config.vocab_size, on_value=0, off_value=1),
          0), [config.beam_width, 1])
  new_prediction_lengths = tf.expand_dims(
      prediction_lengths, 1) + tf.expand_dims(
          tf.to_int32(tf.logical_not(previously_finished)), 1) * lengths_to_add

  # Calculate the scores for each beam
  scores = config.score_fn(total_probs, new_prediction_lengths)

  scores_flat = tf.reshape(scores, [-1])
  # During the first time step we only consider the initial beam
  scores_flat = tf.cond(time_ > 1, lambda: scores_flat, lambda: scores[0])

  # Pick the next beams according to the specified successors function
  next_beam_scores, word_indices = config.choose_successors_fn(scores_flat,
                                                               config)
  next_beam_scores.set_shape([config.beam_width])
  word_indices.set_shape([config.beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  next_beam_probs = tf.gather(total_probs_flat, word_indices)
  next_beam_probs.set_shape([config.beam_width])
  next_word_ids = tf.mod(word_indices, config.vocab_size)
  next_beam_ids = tf.div(word_indices, config.vocab_size)

  # Append new ids to current predictions
  next_predictions = tf.gather(beam_state.predicted_ids, next_beam_ids)
  next_predictions = tf.concat_v2([
      next_predictions[:, 0:time_ - 1],
      tf.to_int32(tf.expand_dims(next_word_ids, 1)), next_predictions[:, time_:]
  ], 1)

  next_beam_state = BeamState(
      time=time_,
      log_probs=next_beam_probs,
      scores=next_beam_scores,
      predicted_ids=next_predictions,
      beam_parent_ids=next_beam_ids)

  return next_beam_state
