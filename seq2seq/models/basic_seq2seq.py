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

from pydoc import locate
import tensorflow as tf
from tensorflow.python.util import nest

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
    self.encoder_class = locate(self.params["encoder.class"])

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
        "encoder.class": "seq2seq.encoders.UnidirectionalRNNEncoder",
        "encoder.params": {}, # Arbitrary parameters for the encoder
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
                    decode_helper,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    # Create Encoder
    enable_dropout = (mode == tf.contrib.learn.ModeKeys.TRAIN)
    encoder_fn = self.encoder_class(self.params["encoder.params"])
    encoder_output = encoder_fn(source, source_len)

    # Create Decoder
    decoder_cell = self._create_decoder_cell(enable_dropout)
    # Define the bridge between encoder and decoder
    bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_cell=decoder_cell)
    decoder_initial_state = bridge()

    max_decode_length = None
    if  mode == tf.contrib.learn.ModeKeys.INFER:
      max_decode_length = self.params["inference.max_decode_length"]

    if self.use_beam_search:
      beam_width = self.params["inference.beam_search.beam_width"]
      decoder_initial_state = nest.map_structure(
          lambda x: tf.tile(x, [beam_width, 1]),
          decoder_initial_state)

    decoder_fn = decoders.BasicDecoder(
        cell=decoder_cell,
        helper=decode_helper,
        initial_state=decoder_initial_state,
        vocab_size=self.target_vocab_info.total_size,
        max_decode_length=max_decode_length)

    if self.use_beam_search:
      decoder_fn = self._get_beam_search_decoder( #pylint: disable=r0204
          decoder_fn)

    decoder_output, final_state = decoder_fn()

    return decoder_output, final_state
