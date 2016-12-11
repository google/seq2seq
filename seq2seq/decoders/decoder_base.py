"""
Base class for sequence decoders.
"""

from collections import namedtuple

import tensorflow as tf
from seq2seq import GraphModule


class DecoderOutput(namedtuple("DecoderOutput", ["logits", "predictions"])):
  """Output of a decoder.

  Note that we output both the logits and predictions because during
  dynamic decoding the predictions may not correspond to max(logits).
  For example, we may be sampling from the logits instead.
  """
  pass


class DecoderStepOutput(
    namedtuple(
        "DecoderStepOutput",
        ["outputs", "next_cell_state", "next_input", "next_loop_state"])):
  """Output of a decoder step to be used with Tensorflow's `raw_rnn`.
  """


class RNNStep(GraphModule):
  """
  A Wrapper around `raw_rnn`.
  """

  def __init__(self,
               step_fn,
               input_fn,
               initial_state,
               sequence_length,
               name="rnn_step"):
    super(RNNStep, self).__init__(name)
    self.step_fn = step_fn
    self.input_fn = input_fn
    self.initial_state = initial_state
    self.sequence_length = sequence_length

  def _build(self, time_, cell_output, cell_state, loop_state):
    if cell_output is None:
      cell_state = self.initial_state

    step_output = self.step_fn(time_, cell_output, cell_state, loop_state,
                               self.input_fn)
    assert isinstance(step_output, DecoderStepOutput), \
      "Step output must be an isntance of DecoderStepOutput"

    if cell_output is None:
      elements_finished = tf.zeros_like(self.sequence_length, dtype=tf.bool)
    else:
      elements_finished = (time_ >= self.sequence_length)

    return (elements_finished, step_output.next_input,
            step_output.next_cell_state, step_output.outputs,
            step_output.next_loop_state)


class FixedDecoderInputs(GraphModule):
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
      self.inputs_ta = self.inputs_ta.unpack(
          tf.transpose(self.inputs, [1, 0, 2]))
      self.max_seq_len = tf.reduce_max(sequence_length, name="max_seq_len")
      self.batch_size = tf.identity(tf.shape(inputs)[0], name="batch_size")
      self.input_dim = tf.identity(tf.shape(inputs)[-1], name="input_dim")

  def _build(self, time_, *args):
    """Returns the input for the given time step.

    Args:
      time_: An int32 scalar

    Returns:
      A tensor of shape `[B, ...]`. When `time_` is past the maximum
      sequence length a zero tensor is fed as input for performance purposes.
    """
    all_finished = (time_ >= self.max_seq_len)
    next_input = tf.cond(
        all_finished,
        lambda: tf.zeros([self.batch_size, self.input_dim], dtype=tf.float32),
        lambda: self.inputs_ta.read(time_))
    next_input.set_shape([None, self.inputs.get_shape().as_list()[-1]])
    return next_input


class DynamicDecoderInputs(GraphModule):
  """An operation that feeds dynamic inputs to a decoder according to some
  arbitrary function that creates a new input from the decoder output at
  the current step, e.g. `embed(argmax(logits))`.

  Args:
    initial_inputs: An input to feed at the first time step.
      A tensor of shape `[B, ...]`.
    make_input_fn: A function that mapes from `(decoder_output) -> next_input`,
      where `next_input` must be a Tensor of shape `[B, ...]`.
  """

  def __init__(self, initial_inputs, make_input_fn,
               name="fixed_decoder_inputs"):
    super(DynamicDecoderInputs, self).__init__(name)
    self.initial_inputs = initial_inputs
    self.make_input_fn = make_input_fn

  def _build(self, _time_, cell_output, _cell_state, _loop_state, step_output):
    """Returns the input for the given time step.
    """
    next_input = self.make_input_fn(step_output)
    if cell_output is None:
      next_input = self.initial_inputs
    return next_input


class DecoderBase(GraphModule):
  """Base class for RNN decoders.

  Args:
    cell: An instance of ` tf.nn.rnn_cell.RNNCell`
    name: A name for this module
  """

  def __init__(self, cell, max_decode_length, name):
    super(DecoderBase, self).__init__(name)
    self.cell = cell
    self.max_decode_length = max_decode_length

  def _step(self, time, cell_output, cell_state, loop_state, next_input_fn):
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
      next_input_fn: A function that generates the next input, e.g. an
        instance of `FixedDecoderInputs` or `DynamicDecoderInputs`.

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

  @staticmethod
  def _pack_outputs(outputs_ta, _final_loop_state):
    """Transposes outputs from time-major to batch-major.
    """
    logits = tf.transpose(outputs_ta.logits.pack(), [1, 0, 2], name="logits")
    predictions = tf.transpose(
        outputs_ta.predictions.pack(), [1, 0], name="predictions")
    return DecoderOutput(logits=logits, predictions=predictions)

  def _build(self, input_fn, initial_state, sequence_length):
    if sequence_length is None:
      sequence_length = self.max_decode_length

    rnn_loop_fn = RNNStep(
        step_fn=self._step,
        input_fn=input_fn,
        initial_state=initial_state,
        sequence_length=tf.minimum(sequence_length, self.max_decode_length))

    outputs_ta, final_state, final_loop_state = tf.nn.raw_rnn(self.cell,
                                                              rnn_loop_fn)
    return self._pack_outputs(outputs_ta,
                              final_loop_state), final_state, final_loop_state
