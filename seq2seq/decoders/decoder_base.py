"""
Base class for sequence decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
from seq2seq.graph_module import GraphModule


class DecoderOutput(namedtuple("DecoderOutput", ["logits", "predicted_ids"])):
  """Output of a decoder.

  Note that we output both the logits and predictions because during
  dynamic decoding the predictions may not correspond to max(logits).
  For example, we may be sampling from the logits instead.
  """
  pass


class DecoderStepOutput(
    namedtuple(
        "DecoderStepOutput",
        ["outputs", "next_cell_state", "next_loop_state"])):
  """Output of a decoder step to be used with Tensorflow's `raw_rnn`.
  """


class RNNStep(GraphModule):
  """
  A Wrapper around `raw_rnn`.
  """

  def __init__(self,
               step_fn,
               next_input_fn,
               initial_state,
               name="rnn_step"):
    super(RNNStep, self).__init__(name)
    self.step_fn = step_fn
    self.next_input_fn = next_input_fn
    self.initial_state = initial_state

  def _build(self, time_, cell_output, cell_state, loop_state):
    initial_call = (cell_output is None)

    if cell_output is None:
      cell_state = self.initial_state

    step_output = self.step_fn(time_, cell_output, cell_state, loop_state)
    next_input, elements_finished = self.next_input_fn(
        time_=time_,
        initial_call=initial_call,
        output=step_output.outputs)

    assert isinstance(step_output, DecoderStepOutput), \
      "Step output must be an isntance of DecoderStepOutput"

    return (elements_finished, next_input,
            step_output.next_cell_state, step_output.outputs,
            step_output.next_loop_state)


class DecoderInputs(GraphModule):
  """Abstract base class for decoder input feeding.
  """

  def __init__(self, name):
    super(DecoderInputs, self).__init__(name)

  def _build(self, time_, initial_call, predicted_ids):
    """Returns the input for the given time step.

    Args:
      time_: An int32 scalar
      initial_call: True iff this is the first time step.
      predicted_ids: The predictions of the decoder. An int32 1-d tensor.

    Returns:
      A tuple of tensors (next_input, finished) where next_input
      is a  a tensor of shape `[B, ...]` and  finished is a boolean tensor
      of shape `[B]`. When `time_` is past the maximum
      sequence length a zero tensor is fed as input for performance purposes.
    """
    raise NotImplementedError

class FixedDecoderInputs(DecoderInputs):
  """An operation that feeds fixed inputs to a decoder,
  also known as "teacher forcing".

  Args:
    inputs: The inputs to feed to the decoder.
      A tensor of shape `[B, T, ...]`. At each time step T, one slice
      of shape `[B, ...]` is fed to the decoder.
    sequence_length: A tensor of shape `[B]` that specifies the
      sequence length for each example.

  """

  def __init__(self, inputs, sequence_length, name="fixed_decoder_inputs"):
    super(FixedDecoderInputs, self).__init__(name)
    self.inputs = inputs
    self.sequence_length = sequence_length

    with self.variable_scope():
      self.inputs_ta = tf.TensorArray(
          dtype=self.inputs.dtype,
          size=tf.shape(self.inputs)[1],
          name="inputs_ta")
      self.inputs_ta = self.inputs_ta.unstack(
          tf.transpose(self.inputs, [1, 0, 2]))
      self.max_seq_len = tf.reduce_max(sequence_length, name="max_seq_len")
      self.batch_size = tf.identity(tf.shape(inputs)[0], name="batch_size")
      self.input_dim = tf.identity(tf.shape(inputs)[-1], name="input_dim")

  def _build(self, time_, initial_call, predicted_ids):
    all_finished = (time_ >= self.max_seq_len)
    next_input = tf.cond(
        all_finished,
        lambda: tf.zeros([self.batch_size, self.input_dim], dtype=tf.float32),
        lambda: self.inputs_ta.read(time_))
    next_input.set_shape([None, self.inputs.get_shape().as_list()[-1]])
    return next_input, (time_ >= self.sequence_length)


class DynamicDecoderInputs(DecoderInputs):
  """An operation that feeds dynamic inputs to a decoder according to some
  arbitrary function that creates a new input from the decoder output at
  the current step, e.g. `embed(argmax(logits))`.

  Args:
    initial_inputs: An input to feed at the first time step.
      A tensor of shape `[B, ...]`.
    make_input_fn: A function that mapes from `predictions -> next_input`,
      where `next_input` must be a Tensor of shape `[B, ...]`.
    max_decode_length: Decode to at most this length
    elements_finished_fn: A function that maps from (time_, predictions) =>
      a boolean vector of shape `[B]` used for early stopping.
  """

  def __init__(self, initial_inputs, make_input_fn,
               max_decode_length,
               elements_finished_fn=None,
               name="fixed_decoder_inputs"):
    super(DynamicDecoderInputs, self).__init__(name)
    self.initial_inputs = initial_inputs
    self.make_input_fn = make_input_fn
    self.max_decode_length = max_decode_length
    self.elements_finished_fn = elements_finished_fn
    self.batch_size = tf.shape(self.initial_inputs)[0]

  def _build(self, time_, initial_call, predictions):
    if initial_call:
      next_input = self.initial_inputs
      elements_finished = tf.zeros([self.batch_size], dtype=tf.bool)
    else:
      next_input = self.make_input_fn(predictions)
      max_decode_length_batch = tf.cast(
          tf.ones([self.batch_size]) * self.max_decode_length,
          dtype=time_.dtype)
      elements_finished = (time_ >= max_decode_length_batch)
      if self.elements_finished_fn:
        elements_finished = tf.logical_or(
            elements_finished, self.elements_finished_fn(time_, predictions))
    return next_input, elements_finished


class DecoderBase(GraphModule):
  """Base class for RNN decoders.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    name: A name for this module
    input_fn: A function that generates the next input, e.g. an
      instance of `FixedDecoderInputs` or `DynamicDecoderInputs`.
  """

  def __init__(self, cell, input_fn, max_decode_length, prediction_fn, name):
    super(DecoderBase, self).__init__(name)
    self.cell = cell
    self.max_decode_length = max_decode_length
    self.input_fn = input_fn

    if prediction_fn is None:
      self.prediction_fn = lambda logits: tf.stop_gradient(tf.argmax(logits, 1))
    else:
      self.prediction_fn = prediction_fn

  def compute_output(self, cell_output):
    """Compute the decoder output based on the current cell state. This method
    should be implemented by all subclasses.

    Args:
      cell_output: The cell outputs for the current time step.
        A float32 tensor of shape `[B, cell.output_size]`

    Returns:
      A (possibly nested) tuple of Tensors that represent decoder-specific
      outputs.
    """
    raise NotImplementedError

  def output_shapes(self):
    """Defines decoder output shapes. Must be implemented by subclasses.

    Returns:
      A (possibly nested) tuple of tensors that defines the output type
      of this decoder. See Tensorflow's raw_rnn initialization
      call for more details.
    """
    raise NotImplementedError

  def create_next_input(self, time_, initial_call, output):
    """Creates the input for the next time step. For decoders that
    do not perform any special input transformations this is a no-op.

    Args:
      time_: The current time step, an int32 scalar
      initial_call: True iff this is the initialization call. In this case
        we want the initial input.
      output: The decoder output at this time step. This is of the same type
        as the return value of `output_shapes`.

    Returns:
      The input for the next time step. A tensor of shape `[batch_size, ...]`.
    """
    return self.input_fn(time_, initial_call, output.predicted_ids)


  def step(self, time_, cell_output, cell_state, loop_state):
    """
    This function maps from the decoder state to the outputs of the current
    time step and the state of the next step. This is where the actual decoding
    logic should be implemented by subclasses.

    The arguments to this function follow those of `tf.nn.raw_rnn`.
    Refer to its documentation for further explanation.

    Args:
      time: An int32 scalar corresponding to the current time step.
      cell_output: The output result of applying the cell function to the input.
        A tensor of shape `[B, cell.output_size]`
      cell_state: The state result of applying the cell function to the input.
        A tensor of shape `[B, cell.state_size]`. This may also be a tuple
        depending on which type of cell is being used.
      loop_state: An optional tuple that can be used to pass state through
        time steps. The shape of this is defined by the subclass.

    Returns:
      A `DecoderStepOutput` tuple, where:

      outputs: The RNN output at this time step. A tuple.
      next_cell_state: The cell state for the next iteration. In most cases
        this is simply the passed in `cell_state`.
        A tensor of shape `[B, cell.state_size]`.
      next_input: The input to the next time step.
        A tensor of shape `[B, ...]`
      next_loop_state: A new loop state of the same type/shape
        as the passed in `loop_state`.
    """
    raise NotImplementedError

  def pack_outputs(self, outputs_ta, _final_loop_state):
    """Transposes outputs from time-major to batch-major.
    """
    logits = outputs_ta.logits.stack()
    predicted_ids = outputs_ta.predicted_ids.stack()
    return DecoderOutput(logits=logits, predicted_ids=predicted_ids)

  def _build(self, initial_state):
    rnn_loop_fn = RNNStep(
        step_fn=self.step,
        next_input_fn=self.create_next_input,
        initial_state=initial_state)

    outputs_ta, final_state, final_loop_state = tf.nn.raw_rnn(
        cell=self.cell,
        loop_fn=rnn_loop_fn,
        swap_memory=True)

    return self.pack_outputs(
        outputs_ta, final_loop_state), final_state, final_loop_state
