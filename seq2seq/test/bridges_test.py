"""
Tests for Encoder-Decoder bridges.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest

from seq2seq.encoders.rnn_encoder import RNNEncoderOutput
from seq2seq.decoders import FixedDecoderInputs, DynamicDecoderInputs
from seq2seq.models.bridges import ZeroBridge, InitialStateBridge
from seq2seq.models.bridges import ConcatInputBridge, PassThroughBridge

class BridgeTest(tf.test.TestCase):
  """Abstract class for bridge tests"""
  def setUp(self):
    super(BridgeTest, self).setUp()
    self.batch_size = 4
    self.sequence_length = 10
    self.input_depth = 10
    self.encoder_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.GRUCell(4), tf.contrib.rnn.GRUCell(8)])
    self.decoder_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.LSTMCell(16), tf.contrib.rnn.GRUCell(8)])
    self.vocab_size = 16
    final_encoder_state = nest.map_structure(
        lambda x: tf.convert_to_tensor(
            value=np.random.randn(self.batch_size, x),
            dtype=tf.float32),
        self.encoder_cell.state_size)
    self.encoder_outputs = RNNEncoderOutput(
        outputs=tf.convert_to_tensor(
            value=np.random.randn(self.batch_size, 10, 16),
            dtype=tf.float32),
        final_state=final_encoder_state)

  def _create_fixed_inputs(self):
    """Creates a FixedDecoderInputs instance"""
    seq_length = tf.ones(
        self.batch_size, dtype=tf.int32) * self.sequence_length
    inputs = tf.convert_to_tensor(
        value=np.random.randn(
            self.batch_size, self.sequence_length, self.input_depth),
        dtype=tf.float32)
    return FixedDecoderInputs(inputs, seq_length)

  def _create_dynamic_inputs(self):
    """Creates a DynamicDecoderInputs instance"""
    with tf.variable_scope("dynamic_inputs",
                           initializer=tf.constant_initializer(0.5)):
      embeddings = tf.get_variable(
          "W_embed", [self.vocab_size, self.input_depth])
      initial_input = tf.random_normal([self.batch_size, self.input_depth])

    def make_input_fn(predicted_ids):
      """Looks up the predictions in the embeddings.
      """
      return tf.nn.embedding_lookup(embeddings, predicted_ids)

    return DynamicDecoderInputs(
        initial_inputs=initial_input,
        make_input_fn=make_input_fn,
        max_decode_length=self.sequence_length)

  def _create_bridge(self, input_fn):
    """Creates the bridge class to be tests. Must be implemented by
    child classes"""
    raise NotImplementedError()

  def _assert_correct_outputs(self):
    """Asserts bridge outputs are correct. Must be implemented by
    child classes"""
    raise NotImplementedError()

  def _run_with_inputs(self, input_fn, **kwargs):
    """Runs the bridge with the given input function and optional
    arguments to be passed to the bridge creation.
    """
    bridge = self._create_bridge(input_fn, **kwargs)
    new_input_fn, initial_state = bridge()
    predicted_ids = np.random.randint(0, self.vocab_size, [self.batch_size])
    orig_input = input_fn(tf.constant(1), False, predicted_ids)
    new_input = new_input_fn(tf.constant(1), False, predicted_ids)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      initial_state_, (new_input_, _) = sess.run(
          [initial_state, new_input])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      orig_input_, _ = sess.run(orig_input)

    return orig_input_, new_input_, initial_state_


class TestZeroBridge(BridgeTest):
  """Tests for the ZeroBridge class"""
  def _create_bridge(self, input_fn, **kwargs):
    return ZeroBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_cell=self.decoder_cell,
        input_fn=input_fn,
        **kwargs)

  def _assert_correct_outputs(self, orig_input_, new_input_, initial_state_):
    initial_state_flat_ = nest.flatten(initial_state_)
    for element in initial_state_flat_:
      np.testing.assert_array_equal(element, np.zeros_like(element))
    np.testing.assert_array_equal(orig_input_, new_input_)

  def test_with_fixed_inputs(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs()))

  def test_with_dynamic_inputs(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs()))

class TestPassThroughBridge(BridgeTest):
  """Tests for the ZeroBridge class"""
  def _create_bridge(self, input_fn, **kwargs):
    return PassThroughBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_cell=self.decoder_cell,
        input_fn=input_fn,
        **kwargs)

  def _assert_correct_outputs(self, orig_input_, new_input_, initial_state_):
    nest.assert_same_structure(initial_state_, self.decoder_cell.state_size)
    nest.assert_same_structure(initial_state_, self.encoder_outputs.final_state)

    encoder_state_flat = nest.flatten(self.encoder_outputs.final_state)
    with self.test_session() as sess:
      encoder_state_flat_ = sess.run(encoder_state_flat)

    initial_state_flat_ = nest.flatten(initial_state_)
    for e_dec, e_enc in zip(initial_state_flat_, encoder_state_flat_):
      np.testing.assert_array_equal(e_dec, e_enc)

    np.testing.assert_array_equal(orig_input_, new_input_)

  def test_with_fixed_inputs(self):
    self.decoder_cell = self.encoder_cell
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs()))

  def test_with_dynamic_inputs(self):
    self.decoder_cell = self.encoder_cell
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs()))


class TestInitialStateBridge(BridgeTest):
  """Tests for the InitialStateBridge class"""
  def _create_bridge(self, input_fn, **kwargs):
    return InitialStateBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_cell=self.decoder_cell,
        input_fn=input_fn,
        **kwargs)

  def _assert_correct_outputs(self, orig_input_, new_input_, initial_state_):
    np.testing.assert_array_equal(orig_input_, new_input_)
    nest.assert_same_structure(initial_state_, self.decoder_cell.state_size)

  def test_with_final_state(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs(),
        bridge_input="final_state",
        activation_fn=None))
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs(),
        bridge_input="final_state",
        activation_fn=None))

  def test_with_outputs(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs(),
        bridge_input="outputs",
        activation_fn=None))
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs(),
        bridge_input="outputs",
        activation_fn=None))

  def test_with_activation_fn(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs(),
        bridge_input="final_state",
        activation_fn="tanh"))
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs(),
        bridge_input="final_state",
        activation_fn="tanh"))



class TestConcatInputBridge(BridgeTest):
  """Tests for the ConcatInputBridge class"""
  def _create_bridge(self, input_fn, **kwargs):
    return ConcatInputBridge(
        encoder_outputs=self.encoder_outputs,
        decoder_cell=self.decoder_cell,
        input_fn=input_fn,
        **kwargs)

  def _assert_correct_outputs(self, _orig_input_, new_input_, initial_state_):
    initial_state_flat_ = nest.flatten(initial_state_)
    for element in initial_state_flat_:
      np.testing.assert_array_equal(element, np.zeros_like(element))

    self.assertEqual(new_input_.shape, (self.batch_size, self.input_depth + 16))

  def test_with_final_state(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs(),
        bridge_input="final_state",
        activation_fn="tanh",
        num_units=16))
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs(),
        bridge_input="final_state",
        activation_fn="tanh",
        num_units=16))

  def test_with_outputs(self):
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_fixed_inputs(),
        bridge_input="outputs",
        activation_fn="tanh",
        num_units=16))
    self._assert_correct_outputs(*self._run_with_inputs(
        input_fn=self._create_dynamic_inputs(),
        bridge_input="outputs",
        activation_fn="tanh",
        num_units=16))

if __name__ == "__main__":
  tf.test.main()
