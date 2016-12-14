"""
Test Cases for Training utils.
"""

import tempfile
import unittest
import tensorflow as tf

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


if __name__ == '__main__':
  unittest.main()
