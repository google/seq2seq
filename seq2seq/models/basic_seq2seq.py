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
from __future__ import unicode_literals

from pydoc import locate
import tensorflow as tf
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

from seq2seq.models.seq2seq_model import Seq2SeqModel
from seq2seq.graph_utils import templatemethod
from seq2seq.models import bridges


class BasicSeq2Seq(Seq2SeqModel):
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

  def __init__(self, params, mode, name="basic_seq2seq"):
    super(BasicSeq2Seq, self).__init__(params, mode, name)
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = Seq2SeqModel.default_params().copy()
    params.update({
        "bridge.class": "seq2seq.models.bridges.InitialStateBridge",
        "bridge.params": {},
        "encoder.class": "seq2seq.encoders.UnidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.BasicDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
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

  def _create_decoder(self, _encoder_output, _features, _labels):
    """Creates a decoder instance based on the passed parameters."""
    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size)

  def _decode_train(self, decoder, bridge, _encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(self.target_embedding,
                                             labels["target_ids"])
    helper_train = tf_decode_helper.TrainingHelper(
        inputs=target_embedded[:, :-1],
        sequence_length=labels["target_len"] - 1)
    decoder_initial_state = bridge()
    return decoder(decoder_initial_state, helper_train)

  def _decode_infer(self, decoder, bridge, _encoder_output, features, labels):
    """Runs decoding in inference mode"""
    batch_size = self.batch_size(features, labels)
    if self.use_beam_search:
      batch_size = self.params["inference.beam_search.beam_width"]

    target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
    helper_infer = tf_decode_helper.GreedyEmbeddingHelper(
        embedding=self.target_embedding,
        start_tokens=tf.fill([batch_size], target_start_id),
        end_token=self.target_vocab_info.special_vocab.SEQUENCE_END)
    decoder_initial_state = bridge()
    return decoder(decoder_initial_state, helper_infer)

  @templatemethod("encode")
  def encode(self, features, labels):
    source_embedded = tf.nn.embedding_lookup(self.source_embedding,
                                             features["source_ids"])
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(source_embedded, features["source_len"])

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    decoder = self._create_decoder(encoder_output, features, labels)
    if self.use_beam_search:
      decoder = self._get_beam_search_decoder(decoder)

    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_state_size=decoder.cell.state_size)
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, bridge, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, bridge, encoder_output, features,
                                labels)
