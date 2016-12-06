"""Featurizers transform the input data into (features, labels) dictionaries
    that can be used with tf.learn model functions.
"""

import tensorflow as tf
import seq2seq

class Seq2SeqFeaturizer(seq2seq.GraphModule):
  def __init__(self, source_vocab_info, target_vocab_info, name="sequence_input"):
    super(Seq2SeqFeaturizer, self).__init__(name)
    self.source_vocab_info = source_vocab_info
    self.target_vocab_info = target_vocab_info

  def _build(self, input_dict):
    output_dict = input_dict.copy()

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, _ = \
      seq2seq.inputs.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, _ = \
      seq2seq.inputs.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Create a graph colleciton for later use
    tf.add_to_collection("source_vocab_to_id", source_vocab_to_id)
    tf.add_to_collection("source_id_to_vocab", source_id_to_vocab)
    tf.add_to_collection("target_vocab_to_id", target_vocab_to_id)
    tf.add_to_collection("target_id_to_vocab", target_id_to_vocab)

    # Look up the source and target in the vocabulary
    output_dict["source_ids"] = source_vocab_to_id.lookup(input_dict["source_tokens"])
    output_dict["target_ids"] = target_vocab_to_id.lookup(input_dict["target_tokens"])

    # Append SEQUENCE_END token to the source
    output_dict["source_ids"] = tf.concat(
      0, [output_dict["source_ids"], [self.source_vocab_info.special_vocab.SEQUENCE_END]])
    output_dict["source_tokens"] = tf.concat(
      0, [output_dict["source_tokens"], ["SEQUENCE_END"]])
    output_dict["source_len"] += 1

    # Prepend SEQUENCE_START token to the target
    output_dict["target_ids"] = tf.concat(
      0, [[self.target_vocab_info.special_vocab.SEQUENCE_START], output_dict["target_ids"]])
    output_dict["target_tokens"] = tf.concat(
      0, [["SEQUENCE_START"], output_dict["target_tokens"]])
    output_dict["target_len"] += 1

    # Cast to int32
    output_dict["source_len"] = tf.to_int32(output_dict["source_len"])
    output_dict["target_len"] = tf.to_int32(output_dict["target_len"])
    output_dict["target_start_id"] = tf.to_int32(self.target_vocab_info.special_vocab.SEQUENCE_START)

    # Separate "features" and "labels"
    features = output_dict
    labels = {}
    labels["target_ids"] = features.pop("target_ids")
    labels["target_tokens"] = features.pop("target_tokens")
    labels["target_len"] = features.pop("target_len")

    return features, labels
