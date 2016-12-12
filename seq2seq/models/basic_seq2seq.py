"""
Definition of a basic seq2seq model
"""

import tensorflow as tf

from seq2seq import training
from seq2seq import encoders
from seq2seq import decoders
from seq2seq.models.model_base import Seq2SeqBase


class BasicSeq2Seq(Seq2SeqBase):
  """Basic Sequence2Sequence model with a unidirectional encoder and decoder.
  The last encoder state is used to initialize the decoder and thus both
  must share the same type of RNN cell.

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
               name="basic_seq2seq"):
    super(BasicSeq2Seq, self).__init__(source_vocab_info, target_vocab_info,
                                       params, name)

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
        "rnn_cell.type": "LSTMCell",
        "rnn_cell.num_units": 128,
        "rnn_cell.dropout_input_keep_prob": 1.0,
        "rnn_cell.dropout_output_keep_prob": 1.0,
        "rnn_cell.num_layers": 1
    })
    return params

  def encode_decode(self,
                    source,
                    source_len,
                    decoder_input_fn,
                    target_len,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    # Create Encoder
    enable_dropout = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    encoder_cell = training.utils.get_rnn_cell(
        cell_type=self.params["rnn_cell.type"],
        num_units=self.params["rnn_cell.num_units"],
        num_layers=self.params["rnn_cell.num_layers"],
        dropout_input_keep_prob=(
            self.params["rnn_cell.dropout_input_keep_prob"]
            if enable_dropout else 1.0),
        dropout_output_keep_prob=(
            self.params["rnn_cell.dropout_output_keep_prob"]
            if enable_dropout else 1.0))
    encoder_fn = encoders.UnidirectionalRNNEncoder(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    # Create Decoder
    # Because we pass the state between encoder and decoder we must
    # use the same cell type
    decoder_cell = encoder_cell
    decoder_fn = decoders.BasicDecoder(
        cell=decoder_cell,
        vocab_size=self.target_vocab_info.total_size,
        max_decode_length=self.params["target.max_seq_len"])

    decoder_output, _, _ = decoder_fn(
        input_fn=decoder_input_fn,
        initial_state=encoder_output.final_state,
        sequence_length=target_len)

    return decoder_output
