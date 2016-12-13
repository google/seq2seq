"""
Unit tests for input-related operations.
"""

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.data import parallel_data_provider
from seq2seq.data import split_tokens_decoder
from seq2seq.data import data_utils
from seq2seq.test import utils as test_utils

class SplitTokensDecoderTest(tf.test.TestCase):
  """Tests the SplitTokensDecoder class
  """
  def test_decode(self):
    decoder = split_tokens_decoder.SplitTokensDecoder(
        delimiter=" ",
        tokens_feature_name="source_tokens",
        length_feature_name="source_len")

    self.assertEqual(
        decoder.list_items(),
        ["source_tokens", "source_len"])

    data = tf.constant("Hello world !")

    decoded_tokens = decoder.decode(data, ["source_tokens"])
    decoded_length = decoder.decode(data, ["source_len"])
    decoded_both = decoder.decode(data, decoder.list_items())

    with self.test_session() as sess:
      decoded_tokens_ = sess.run(decoded_tokens)[0]
      decoded_length_ = sess.run(decoded_length)[0]
      decoded_both_ = sess.run(decoded_both)

    self.assertEqual(decoded_length_, 3)
    np.testing.assert_array_equal(
        decoded_tokens_.astype("U"),
        ["Hello", "world", "!"])

    self.assertEqual(decoded_both_[1], 3)
    np.testing.assert_array_equal(
        decoded_both_[0].astype("U"),
        ["Hello", "world", "!"])


class ParallelDataProviderTest(tf.test.TestCase):
  """Tests the ParallelDataProvider class
  """
  def setUp(self):
    super(ParallelDataProviderTest, self).setUp()
    # Our data
    self.source_lines = ["Hello", "World", "!"]
    self.target_lines = ["1", "2", "3"]
    self.source_to_target = dict(zip(self.source_lines, self.target_lines))

    # Create two parallel text files
    self.source_file = tempfile.NamedTemporaryFile("w")
    self.target_file = tempfile.NamedTemporaryFile("w")
    self.source_file.write("\n".join(self.source_lines))
    self.source_file.flush()
    self.target_file.write("\n".join(self.target_lines))
    self.target_file.flush()

  def tearDown(self):
    super(ParallelDataProviderTest, self).tearDown()
    self.source_file.close()
    self.target_file.close()

  def test_reading(self):
    num_epochs = 50
    data_provider = data_utils.make_parallel_data_provider(
        data_sources_source=[self.source_file.name],
        data_sources_target=[self.target_file.name],
        num_epochs=num_epochs,
        shuffle=True)

    item_keys = list(data_provider.list_items())
    item_values = data_provider.get(item_keys)
    items_dict = dict(zip(item_keys, item_values))

    self.assertEqual(
        set(item_keys),
        set(["source_tokens", "source_len", "target_tokens", "target_len"]))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        item_dicts_ = [sess.run(items_dict) for _ in range(num_epochs * 3)]

    for item_dict in item_dicts_:
      self.assertEqual(item_dict["source_len"], 1)
      self.assertEqual(item_dict["target_len"], 1)
      item_dict["target_tokens"] = item_dict["target_tokens"].astype("U")
      item_dict["source_tokens"] = item_dict["source_tokens"].astype("U")
      self.assertEqual(
          item_dict["target_tokens"][0],
          self.source_to_target[item_dict["source_tokens"][0]])


class ReadFromDataProviderTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(ReadFromDataProviderTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_read_from_data_provider(self):
    file = test_utils.create_temp_tfrecords(
        source="Hello World .", target="Bye")
    data_provider = data_utils.make_tfrecord_data_provider(
        [file.name], num_epochs=5)
    features = data_utils.read_from_data_provider(data_provider)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        res = sess.run(features)

    self.assertEqual(res["source_len"], 3)
    self.assertEqual(res["target_len"], 1)
    np.testing.assert_array_equal(res["source_tokens"].astype("U"),
                                  ["Hello", "World", "."])
    np.testing.assert_array_equal(res["target_tokens"].astype("U"), ["Bye"])


if __name__ == "__main__":
  tf.test.main()
