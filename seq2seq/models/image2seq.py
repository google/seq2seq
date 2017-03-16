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

import tensorflow as tf

from seq2seq import graph_utils
from seq2seq.data import vocab
from seq2seq.graph_utils import templatemethod
from seq2seq.models.model_base import ModelBase
from seq2seq.models.attention_seq2seq import AttentionSeq2Seq


class Image2Seq(AttentionSeq2Seq):
  """A model that encodes an image and produces a sequence
  of tokens.
  """

  def __init__(self, params, mode, name="image_seq2seq"):
    super(Image2Seq, self).__init__(params, mode, name)
    self.params["source.reverse"] = False
    self.params["embedding.share"] = False

  @staticmethod
  def default_params():
    params = ModelBase.default_params()
    params.update({
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {
            "num_units": 128
        },
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "bridge.params": {},
        "encoder.class": "seq2seq.encoders.InceptionV3Encoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {},  # Arbitrary parameters for the decoder
        "target.max_seq_len": 50,
        "embedding.dim": 100,
        "inference.beam_search.beam_width": 0,
        "inference.beam_search.length_penalty_weight": 0.0,
        "inference.beam_search.choose_successors_fn": "choose_top_k",
        "vocab_target": "",
    })
    return params

  @templatemethod("encode")
  def encode(self, features, _labels):
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(features["image"])

  def batch_size(self, features, _labels):
    return tf.shape(features["image"])[0]

  def _preprocess(self, features, labels):
    """Model-specific preprocessing for features and labels:

    - Creates vocabulary lookup tables for target vocab
    - Converts tokens into vocabulary ids
    - Prepends a speical "SEQUENCE_START" token to the target
    - Appends a speical "SEQUENCE_END" token to the target
    """

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    if labels is None:
      return features, None

    labels = labels.copy()

    # Slices targets to max length
    if self.params["target.max_seq_len"] is not None:
      labels["target_tokens"] = labels["target_tokens"][:, :self.params[
          "target.max_seq_len"]]
      labels["target_len"] = tf.minimum(labels["target_len"],
                                        self.params["target.max_seq_len"])

    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    return features, labels
