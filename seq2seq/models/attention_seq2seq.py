# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sequence to Sequence model with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

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
               mode,
               name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(
        source_vocab_info, target_vocab_info, params, mode, name)

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.dim": 128,
        "attention.score_type": "dot",
        "bridge_spec": {
            "class": "ZeroBridge",
        },
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {}, # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {} # Arbitrary parameters for the decoder
    })
    return params

  def encode_decode(self,
                    source,
                    source_len,
                    decode_helper):
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    encoder_output = encoder_fn(source, source_len)

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

    max_decode_length = None
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      max_decode_length = self.params["inference.max_decode_length"]

    decoder_fn = self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        max_decode_length=max_decode_length,
        attention_inputs=encoder_output.outputs,
        attention_inputs_length=source_len,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths)

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    # Bridge between encoder and decoder
    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_cell=decoder_fn.cell)
    decoder_initial_state = bridge()

    if self.use_beam_search:
      beam_width = self.params["inference.beam_search.beam_width"]
      decoder_initial_state = nest.map_structure(
          lambda x: tf.tile(x, [beam_width, 1]),
          decoder_initial_state)

    decoder_output, final_state = decoder_fn(
        decoder_initial_state, decode_helper)

    return decoder_output, final_state
