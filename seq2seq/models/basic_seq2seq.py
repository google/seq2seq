"""
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
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
    assert hasattr(encoders, params["encoder.type"]), (
        "Invalid encoder type: {}".format(params["encoder.type"]))
    self.encoder_class = getattr(encoders, params["encoder.type"])

  def _create_encoder_cell(self, enable_dropout):
    """Creates a cell instance for the encoder based on the model parameters"""
    return training.utils.get_rnn_cell(
        cell_spec=self.params["encoder.rnn_cell.cell_spec"],
        num_layers=self.params["encoder.rnn_cell.num_layers"],
        dropout_input_keep_prob=(
            self.params["encoder.rnn_cell.dropout_input_keep_prob"]
            if enable_dropout else 1.0),
        dropout_output_keep_prob=(
            self.params["encoder.rnn_cell.dropout_output_keep_prob"]
            if enable_dropout else 1.0),
        residual_connections=self.params[
            "encoder.rnn_cell.residual_connections"],
        residual_combiner=self.params["encoder.rnn_cell.residual_combiner"],
        residual_dense=self.params["encoder.rnn_cell.residual_dense"])

  def _create_decoder_cell(self, enable_dropout):
    """Creates a cell instance for the decoder based on the model parameters"""
    return training.utils.get_rnn_cell(
        cell_spec=self.params["decoder.rnn_cell.cell_spec"],
        num_layers=self.params["decoder.rnn_cell.num_layers"],
        dropout_input_keep_prob=(
            self.params["decoder.rnn_cell.dropout_input_keep_prob"]
            if enable_dropout else 1.0),
        dropout_output_keep_prob=(
            self.params["decoder.rnn_cell.dropout_output_keep_prob"]
            if enable_dropout else 1.0),
        residual_connections=self.params[
            "decoder.rnn_cell.residual_connections"],
        residual_combiner=self.params["decoder.rnn_cell.residual_combiner"],
        residual_dense=self.params["decoder.rnn_cell.residual_dense"])

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
        "bridge_spec": {
            "class": "InitialStateBridge",
        },
        "encoder.type": "UnidirectionalRNNEncoder",
        "encoder.rnn_cell.cell_spec": {
            "class": "BasicLSTMCell",
            "num_units": 128
        },
        "encoder.rnn_cell.dropout_input_keep_prob": 1.0,
        "encoder.rnn_cell.dropout_output_keep_prob": 1.0,
        "encoder.rnn_cell.num_layers": 1,
        "encoder.rnn_cell.residual_connections": False,
        "encoder.rnn_cell.residual_combiner": "add",
        "encoder.rnn_cell.residual_dense": False,
        "decoder.rnn_cell.cell_spec": {
            "class": "BasicLSTMCell",
            "num_units": 128
        },
        "decoder.rnn_cell.dropout_input_keep_prob": 1.0,
        "decoder.rnn_cell.dropout_output_keep_prob": 1.0,
        "decoder.rnn_cell.num_layers": 1,
        "decoder.rnn_cell.residual_connections": False,
        "decoder.rnn_cell.residual_combiner": "add",
        "decoder.rnn_cell.residual_dense": False
    })
    return params

  def encode_decode(self,
                    source,
                    source_len,
                    decoder_input_fn,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    # Create Encoder
    enable_dropout = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    encoder_cell = self._create_encoder_cell(enable_dropout)
    encoder_fn = self.encoder_class(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    # Create Decoder
    decoder_cell = self._create_decoder_cell(enable_dropout)
    # Define the bridge between encoder and decoder
    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_cell=decoder_cell,
        input_fn=decoder_input_fn)
    new_decoder_input_fn, decoder_initial_state = bridge()

    decoder_fn = decoders.BasicDecoder(
        cell=decoder_cell,
        input_fn=new_decoder_input_fn,
        vocab_size=self.target_vocab_info.total_size,
        max_decode_length=self.params["inference.max_decode_length"])

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    decoder_output, _, _ = decoder_fn(
        initial_state=decoder_initial_state)

    return decoder_output
