"""
Sequence to Sequence model with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq import decoders
from seq2seq.models.basic_seq2seq import BasicSeq2Seq


class AttentionSeq2Seq(BasicSeq2Seq):
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
    super(AttentionSeq2Seq, self).__init__(
        source_vocab_info, target_vocab_info, params, name)

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.dim": 128,
        "attention.score_type": "dot",
        "bridge_spec": {
            "class": "ZeroBridge",
        },
    })
    return params

  def encode_decode(self,
                    source,
                    source_len,
                    decoder_input_fn,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    enable_dropout = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    encoder_cell = self._create_encoder_cell(enable_dropout)
    encoder_fn = self.encoder_class(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    decoder_cell = self._create_decoder_cell(enable_dropout)
    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_cell=decoder_cell,
        input_fn=decoder_input_fn)
    new_decoder_input_fn, decoder_initial_state = bridge()
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
        input_fn=new_decoder_input_fn,
        vocab_size=self.target_vocab_info.total_size,
        attention_inputs=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths,
        max_decode_length=self.params["inference.max_decode_length"])

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    decoder_output, _, _ = decoder_fn(
        initial_state=decoder_initial_state)

    return decoder_output
