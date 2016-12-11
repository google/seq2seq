"""
Sequence to Sequence model with attention
"""

import tensorflow as tf

from seq2seq import encoders
from seq2seq import decoders
from seq2seq.training import utils as training_utils
from seq2seq.models.model_base import Seq2SeqBase


class AttentionSeq2Seq(Seq2SeqBase):
  """Sequence2Sequence model with attention mechanism.

  Args:
    source_vocab_info: An instance of `seq2seq.inputs.VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `seq2seq.inputs.VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self,
               source_vocab_info,
               target_vocab_info,
               params,
               name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(source_vocab_info, target_vocab_info,
                                           params, name)

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
        "attention.dim": 128,
        "rnn_cell.type": "LSTMCell",
        "rnn_cell.num_units": 128,
        "rnn_cell.dropout_input_keep_prob": 1.0,
        "rnn_cell.dropout_output_keep_prob": 1.0,
        "rnn_cell.num_layers": 1
    })
    return params

  def encode_decode(self, source, source_len, decoder_input_fn, target_len):
    encoder_cell = training_utils.get_rnn_cell(
        cell_type=self.params["rnn_cell.type"],
        num_units=self.params["rnn_cell.num_units"],
        num_layers=self.params["rnn_cell.num_layers"],
        dropout_input_keep_prob=self.params["rnn_cell.dropout_input_keep_prob"],
        dropout_output_keep_prob=self.params[
            "rnn_cell.dropout_output_keep_prob"])
    encoder_fn = encoders.BidirectionalRNNEncoder(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    decoder_cell = encoder_cell
    decoder_fn = decoders.AttentionDecoder(
        cell=decoder_cell,
        vocab_size=self.target_vocab_info.total_size,
        attention_inputs=encoder_output.outputs,
        attention_fn=decoders.AttentionLayer(self.params["attention.dim"]),
        max_decode_length=self.params["target.max_seq_len"])

    decoder_output, _, _ = decoder_fn(
        input_fn=decoder_input_fn,
        initial_state=decoder_cell.zero_state(
            tf.shape(source_len)[0], dtype=tf.float32),
        sequence_length=target_len)

    return decoder_output
