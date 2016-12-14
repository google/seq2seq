"""
Test Cases for Training utils.
"""

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.training import utils as training_utils
from seq2seq.training import HParamsParser


class TestHparamsReadWrite(tf.test.TestCase):
  """Tests reading and writing of HParam values
  """

  def setUp(self):
    super(TestHparamsReadWrite, self).setUp()
    self.hparams = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    self.parser = HParamsParser(self.hparams)

  def test_write(self):
    file = tempfile.NamedTemporaryFile()
    training_utils.write_hparams(self.hparams, file.name)
    lines = file.read()
    self.assertEqual(
        lines.decode(),
        "dropout=0.8\nnum_layers=2\nrnn_cell_type=LSTM\nrnn_dim=128")
    file.close()

  def test_write_read(self):
    file = tempfile.NamedTemporaryFile()
    training_utils.write_hparams(self.hparams, file.name)
    params_str = training_utils.read_hparams(file.name)
    final_params = self.parser.parse(params_str)
    self.assertEqual(final_params, self.hparams)


class TestLRDecay(tf.test.TestCase):
  """Tests learning rate decay function.
  """

  def test_no_decay(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type=None,
        decay_steps=5,
        decay_rate=2.0)
    self.assertEqual(decay_fn, None)

    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="",
        decay_steps=5,
        decay_rate=2.0)
    self.assertEqual(decay_fn, None)

  def test_decay_without_min(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="exponential_decay",
        decay_steps=10,
        decay_rate=0.9,
        start_decay_at=100,
        stop_decay_at=1000,
        staircase=False)

    initial_lr = 1.0
    with self.test_session() as sess:
      # Should not decay before start_decay_at
      np.testing.assert_equal(
          sess.run(decay_fn(initial_lr, 50)),
          initial_lr)
      # Proper decay
      np.testing.assert_almost_equal(
          sess.run(decay_fn(initial_lr, 115)),
          initial_lr * 0.9**(15.0 / 10.0))
      # Should not decay past stop_decay_at
      np.testing.assert_almost_equal(
          sess.run(decay_fn(initial_lr, 5000)),
          initial_lr * 0.9**((1000.0-100.0) / 10.0))


  def test_decay_with_min(self):
    decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type="exponential_decay",
        decay_steps=10,
        decay_rate=0.9,
        start_decay_at=100,
        stop_decay_at=1000,
        min_learning_rate=0.01,
        staircase=False)

    initial_lr = 1.0
    with self.test_session() as sess:
      # Should not decay past min_learning_rate
      np.testing.assert_almost_equal(
          sess.run(decay_fn(initial_lr, 900)),
          0.01)

if __name__ == '__main__':
  tf.test.main()
