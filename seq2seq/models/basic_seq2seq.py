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
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from pydoc import locate
import tensorflow as tf

from seq2seq.models.model_base import Seq2SeqBase
from seq2seq.models import bridges


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
               mode,
               name="basic_seq2seq"):
    super(BasicSeq2Seq, self).__init__(source_vocab_info, target_vocab_info,
                                       params, mode, name)
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
        "bridge.class": "seq2seq.models.bridges.InitialStateBridge",
        "bridge.params": {},
        "encoder.class": "seq2seq.encoders.UnidirectionalRNNEncoder",
        "encoder.params": {}, # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.BasicDecoder",
        "decoder.params": {} # Arbitrary parameters for the decoder
    })
    return params

  def _create_bridge(self, encoder_outputs, decoder_state_size):
    """Creates the bridge to be used between encoder and decoder"""
    bridge_class = locate(self.params["bridge.class"]) or \
      getattr(bridges, self.params["bridge.class"])
    return bridge_class(
        encoder_outputs=encoder_outputs,
        decoder_state_size=decoder_state_size,
        params=self.params["bridge.params"],
        mode=self.mode)

  def _create_encoder(self, _source, _source_len):
    """Creates the encoder function for this model"""
    return self.encoder_class(self.params["encoder.params"], self.mode)

  def _create_decoder(self, _encoder_output, _source, _source_len):
    """Creates the decoder function for this model"""
    max_decode_length = None
    if  self.mode == tf.contrib.learn.ModeKeys.INFER:
      max_decode_length = self.params["inference.max_decode_length"]

    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        max_decode_length=max_decode_length)

  def encode_decode(self,
                    source,
                    source_len,
                    decode_helper):
    # Create Encoder
    encoder_fn = self._create_encoder(source, source_len)
    encoder_output = encoder_fn(source, source_len)
    decoder_fn = self._create_decoder(encoder_output, source, source_len)

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    # Bridge between encoder and decoder
    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_state_size=decoder_fn.cell.state_size)
    decoder_initial_state = bridge()

    decoder_output, final_state = decoder_fn(
        decoder_initial_state, decode_helper)

    return decoder_output, final_state, encoder_output
