"""
Unit tests for HParamsParser.
"""

import unittest
from seq2seq.training import HParamsParser


class HParamsParserTest(unittest.TestCase):
  """Test for HParamsParser class.
  """

  def test_parse(self):
    default_params = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    parser = HParamsParser(default_params)
    final_params = parser.parse("rnn_dim=256,rnn_cell_type=GRU,dropout=0.77")
    self.assertEqual(final_params["rnn_dim"], 256)
    self.assertEqual(final_params["rnn_cell_type"], "GRU")
    self.assertEqual(final_params["dropout"], 0.77)
    self.assertEqual(final_params["num_layers"], 2)

  def test_parse_with_newlines(self):
    default_params = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    parser = HParamsParser(default_params)
    final_params = parser.parse("\n".join(
        ["rnn_dim=256,", "rnn_cell_type=GRU,", "dropout=0.77"]))
    self.assertEqual(final_params["rnn_dim"], 256)
    self.assertEqual(final_params["rnn_cell_type"], "GRU")
    self.assertEqual(final_params["dropout"], 0.77)
    self.assertEqual(final_params["num_layers"], 2)


if __name__ == '__main__':
  unittest.main()
