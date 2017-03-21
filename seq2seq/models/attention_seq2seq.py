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
from __future__ import unicode_literals

from pydoc import locate

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

  def __init__(self, params, mode, name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(params, mode, name)

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params

  def _create_decoder(self, encoder_output, features, _labels):
    attention_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])
    attention_layer = attention_class(
        params=self.params["attention.params"], mode=self.mode)

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        attention_values=encoder_output.attention_values,
        attention_values_length=encoder_output.attention_values_length,
        attention_keys=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths)
