"""Featurizers transform the input data into (features, labels) dictionaries
    that can be used with tf.learn model functions.
"""

import tensorflow as tf
from seq2seq import inputs
from seq2seq.graph_module import GraphModule


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
    source_vocab_info: a `seq2seq.inputs.VocabInfo` for the source vocab
    source_vocab_info: a `seq2seq.inputs.VocabInfo` for the target vocab
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

  def _build(self, input_dict):
    output_dict = input_dict.copy()

    # TODO: Ideally we should have the "special vocabulary" in our lookup table.
    # How to best do this? Create a temporary files with the special vocab?

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, _ = \
      inputs.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, _ = \
      inputs.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Create a graph colleciton for later use
    # TODO: Is there a nicer way to do this?
    # See https://github.com/dennybritz/seq2seq/issues/21
    tf.add_to_collection("source_vocab_to_id", source_vocab_to_id)
    tf.add_to_collection("source_id_to_vocab", source_id_to_vocab)
    tf.add_to_collection("target_vocab_to_id", target_vocab_to_id)
    tf.add_to_collection("target_id_to_vocab", target_id_to_vocab)

    if self.max_seq_len_source is not None:
      output_dict["source_tokens"] = output_dict[
          "source_tokens"][:self.max_seq_len_source - 1]
      output_dict["source_len"] = tf.minimum(output_dict["source_len"],
                                             self.max_seq_len_source - 1)
    if self.max_seq_len_target is not None:
      output_dict["target_tokens"] = output_dict[
          "target_tokens"][:self.max_seq_len_target - 2]
      output_dict["target_len"] = tf.minimum(output_dict["target_len"],
                                             self.max_seq_len_target - 2)

    # Look up the source and target in the vocabulary
    output_dict["source_ids"] = source_vocab_to_id.lookup(output_dict[
        "source_tokens"])
    output_dict["target_ids"] = target_vocab_to_id.lookup(output_dict[
        "target_tokens"])

    # Append SEQUENCE_END token to the source
    output_dict["source_ids"] = tf.concat(0, [
        output_dict["source_ids"],
        [self.source_vocab_info.special_vocab.SEQUENCE_END]
    ])
    output_dict["source_tokens"] = tf.concat(
        0, [output_dict["source_tokens"], ["SEQUENCE_END"]])
    output_dict["source_len"] += 1

    # Prepend SEQUENCE_START token to the target
    output_dict["target_ids"] = tf.concat(
        0, [[self.target_vocab_info.special_vocab.SEQUENCE_START],
            output_dict["target_ids"]])
    output_dict["target_tokens"] = tf.concat(
        0, [["SEQUENCE_START"], output_dict["target_tokens"]])
    output_dict["target_len"] += 1

    # Append SEQUENCE_END token to the target
    output_dict["target_ids"] = tf.concat(0, [
        output_dict["target_ids"],
        [self.target_vocab_info.special_vocab.SEQUENCE_END]
    ])
    output_dict["target_tokens"] = tf.concat(
        0, [output_dict["target_tokens"], ["SEQUENCE_END"]])
    output_dict["target_len"] += 1

    # Cast to int32
    output_dict["source_len"] = tf.to_int32(output_dict["source_len"])
    output_dict["target_len"] = tf.to_int32(output_dict["target_len"])
    output_dict["target_start_id"] = tf.to_int32(
        self.target_vocab_info.special_vocab.SEQUENCE_START)
    output_dict["target_end_id"] = tf.to_int32(
        self.target_vocab_info.special_vocab.SEQUENCE_END)

    # Add summaries
    tf.summary.histogram("source_len", output_dict["source_len"])
    tf.summary.histogram("target_len", output_dict["target_len"])

    # Separate "features" and "labels"
    features = output_dict
    labels = {}
    labels["target_ids"] = features.pop("target_ids")
    labels["target_tokens"] = features.pop("target_tokens")
    labels["target_len"] = features.pop("target_len")

    return features, labels
