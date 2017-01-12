"""
Sequence to Sequence model with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq import encoders
from seq2seq import decoders
from seq2seq.training import utils as training_utils
from seq2seq.models.model_base import Seq2SeqBase


class AttentionSeq2Seq(Seq2SeqBase):
  """Sequence2Sequence model with attention mechanism.

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
               name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(source_vocab_info, target_vocab_info,
                                           params, name)

    assert hasattr(encoders, params["encoder.type"]), (
        "Invalid encoder type: {}".format(params["encoder.type"]))
    self.encoder_class = getattr(encoders, params["encoder.type"])

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
        "attention.dim": 128,
        "attention.score_type": "dot",
        "encoder.type": "UnidirectionalRNNEncoder",
        "encoder.rnn_cell.type": "BasicLSTMCell",
        "encoder.rnn_cell.num_units": 128,
        "encoder.rnn_cell.dropout_input_keep_prob": 1.0,
        "encoder.rnn_cell.dropout_output_keep_prob": 1.0,
        "encoder.rnn_cell.num_layers": 1,
        "encoder.rnn_cell.residual_connections": False,
        "encoder.rnn_cell.extra_args_json": "",
        "decoder.rnn_cell.type": "BasicLSTMCell",
        "decoder.rnn_cell.num_units": 128,
        "decoder.rnn_cell.dropout_input_keep_prob": 1.0,
        "decoder.rnn_cell.dropout_output_keep_prob": 1.0,
        "decoder.rnn_cell.num_layers": 1,
        "decoder.rnn_cell.residual_connections": False,
        "encoder.rnn_cell.extra_args_json": ""
    })
    return params

  def encode_decode(self,
                    source,
                    source_len,
                    decoder_input_fn,
                    target_len,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    enable_dropout = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    encoder_cell = training_utils.get_rnn_cell(
        cell_type=self.params["encoder.rnn_cell.type"],
        num_units=self.params["encoder.rnn_cell.num_units"],
        num_layers=self.params["encoder.rnn_cell.num_layers"],
        dropout_input_keep_prob=(
            self.params["encoder.rnn_cell.dropout_input_keep_prob"]
            if enable_dropout else 1.0),
        dropout_output_keep_prob=(
            self.params["encoder.rnn_cell.dropout_output_keep_prob"]
            if enable_dropout else 1.0),
        residual_connections=self.params[
            "encoder.rnn_cell.residual_connections"],
        extra_args_json=self.params["encoder.rnn_cell.extra_args_json"])
    encoder_fn = self.encoder_class(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    decoder_cell = training_utils.get_rnn_cell(
        cell_type=self.params["decoder.rnn_cell.type"],
        num_units=self.params["decoder.rnn_cell.num_units"],
        num_layers=self.params["decoder.rnn_cell.num_layers"],
        dropout_input_keep_prob=(
            self.params["decoder.rnn_cell.dropout_input_keep_prob"]
            if enable_dropout else 1.0),
        dropout_output_keep_prob=(
            self.params["decoder.rnn_cell.dropout_output_keep_prob"]
            if enable_dropout else 1.0),
        residual_connections=self.params[
            "decoder.rnn_cell.residual_connections"],
        extra_args_json=self.params["decoder.rnn_cell.extra_args_json"])

    attention_layer = decoders.AttentionLayer(
        num_units=self.params["attention.dim"],
        score_type=self.params["attention.score_type"])

    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = source_len
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    decoder_fn = decoders.AttentionDecoder(
        cell=decoder_cell,
        input_fn=decoder_input_fn,
        vocab_size=self.target_vocab_info.total_size,
        attention_inputs=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths,
        max_decode_length=self.params["target.max_seq_len"])

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    decoder_output, _, _ = decoder_fn(
        initial_state=decoder_cell.zero_state(
            tf.shape(source_len)[0], dtype=tf.float32),
        sequence_length=target_len)

    return decoder_output
