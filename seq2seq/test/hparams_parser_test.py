"""
Unit tests for HParamsParser.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        "dropout": 0.8,
        "bool": True
    }
    parser = HParamsParser(default_params)
    final_params = parser.parse(
        "rnn_dim=256,rnn_cell_type=GRU,dropout=0.77,bool=False")
    self.assertEqual(final_params["rnn_dim"], 256)
    self.assertEqual(final_params["rnn_cell_type"], "GRU")
    self.assertEqual(final_params["dropout"], 0.77)
    self.assertEqual(final_params["num_layers"], 2)
    self.assertEqual(final_params["bool"], False)

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

  def test_parse_without_defaults(self):
    default_params = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    parser = HParamsParser(default_params)
    final_params = parser.parse(
        "\n".join(["rnn_dim=256,", "rnn_cell_type=GRU,", "dropout=0.77"]),
        with_defaults=False)
    self.assertEqual(final_params["rnn_dim"], 256)
    self.assertEqual(final_params["rnn_cell_type"], "GRU")
    self.assertEqual(final_params["dropout"], 0.77)
    self.assertNotIn("num_layers", final_params)

  def test_parse_with_float(self):
    default_params = {
        "rnn_dim": 128,
        "num_layers": 2,
        "rnn_cell_type": "LSTM",
        "dropout": 0.8
    }
    parser = HParamsParser(default_params)
    final_params = parser.parse(
        "\n".join(["rnn_dim=256.00,", "rnn_cell_type=GRU,", "dropout=0.77"]))
    self.assertEqual(final_params["rnn_dim"], 256)
    self.assertEqual(final_params["rnn_cell_type"], "GRU")
    self.assertEqual(final_params["dropout"], 0.77)
    self.assertEqual(final_params["num_layers"], 2)


if __name__ == '__main__':
  unittest.main()
