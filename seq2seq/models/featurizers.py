"""Featurizers transform the input data into (features, labels) dictionaries
    that can be used with tf.learn model functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from seq2seq.data import vocab
from seq2seq.graph_module import GraphModule
from seq2seq import graph_utils


class Seq2SeqFeaturizer(GraphModule):
  """Takes raw tensors read from a TFRecods file and transforms them into
  feature and labels dictionaries that can be fed to model functions.
  In particular, this featurizer:

  - Creates vocabulary lookup tables for source and target vocab
  - Converts tokens into vocabulary ids
  - Appends a speical "SEQUENCE_END" token to the source
  - Prepends a speical "SEQUENCE_START" token to the target
  - Appends a speical "SEQUENCE_END" token to the target

  Args:
    source_vocab_info: a `VocabInfo` for the source vocab
    source_vocab_info: a `VocabInfo` for the target vocab
  """

  def __init__(self,
               source_vocab_info,
               target_vocab_info,
               max_seq_len_source=None,
               max_seq_len_target=None,
               name="sequence_input"):
    super(Seq2SeqFeaturizer, self).__init__(name)
    self.source_vocab_info = source_vocab_info
    self.target_vocab_info = target_vocab_info
    self.max_seq_len_source = max_seq_len_source
    self.max_seq_len_target = max_seq_len_target

  def _build(self, features, labels):
    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, _ = \
      vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, _ = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "source_vocab_to_id": source_vocab_to_id,
        "source_id_to_vocab": source_id_to_vocab,
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab
    }, "vocab_tables")

    # Slice source to max_len
    if self.max_seq_len_source is not None:
      features["source_tokens"] = features[
          "source_tokens"][:, :self.max_seq_len_source]
      features["source_len"] = tf.minimum(
          features["source_len"], self.max_seq_len_source)

    # Look up the source ids in the vocabulary
    features["source_ids"] = source_vocab_to_id.lookup(features[
        "source_tokens"])

    features["source_len"] = tf.to_int32(features["source_len"])
    tf.summary.histogram("source_len", tf.to_float(features["source_len"]))

    if labels is None:
      return features, None

    labels = labels.copy()

    # Slices targets to max length
    if self.max_seq_len_target is not None:
      labels["target_tokens"] = labels[
          "target_tokens"][:, :self.max_seq_len_target]
      labels["target_len"] = tf.minimum(
          labels["target_len"], self.max_seq_len_target)

    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    return features, labels
