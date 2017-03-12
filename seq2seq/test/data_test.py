# -*- coding: utf-8 -*-
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
Unit tests for input-related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.data import split_tokens_decoder
from seq2seq.data.parallel_data_provider import make_parallel_data_provider


class SplitTokensDecoderTest(tf.test.TestCase):
  """Tests the SplitTokensDecoder class
  """

  def test_decode(self):
    decoder = split_tokens_decoder.SplitTokensDecoder(
        delimiter=" ",
        tokens_feature_name="source_tokens",
        length_feature_name="source_len")

    self.assertEqual(decoder.list_items(), ["source_tokens", "source_len"])

    data = tf.constant("Hello world ! 笑ｗ")

    decoded_tokens = decoder.decode(data, ["source_tokens"])
    decoded_length = decoder.decode(data, ["source_len"])
    decoded_both = decoder.decode(data, decoder.list_items())

    with self.test_session() as sess:
      decoded_tokens_ = sess.run(decoded_tokens)[0]
      decoded_length_ = sess.run(decoded_length)[0]
      decoded_both_ = sess.run(decoded_both)

    self.assertEqual(decoded_length_, 4)
    np.testing.assert_array_equal(
        np.char.decode(decoded_tokens_.astype("S"), "utf-8"),
        ["Hello", "world", "!", "笑ｗ"])

    self.assertEqual(decoded_both_[1], 4)
    np.testing.assert_array_equal(
        np.char.decode(decoded_both_[0].astype("S"), "utf-8"),
        ["Hello", "world", "!", "笑ｗ"])


class ParallelDataProviderTest(tf.test.TestCase):
  """Tests the ParallelDataProvider class
  """

  def setUp(self):
    super(ParallelDataProviderTest, self).setUp()
    # Our data
    self.source_lines = ["Hello", "World", "!", "笑"]
    self.target_lines = ["1", "2", "3", "笑"]
    self.source_to_target = dict(zip(self.source_lines, self.target_lines))

    # Create two parallel text files
    self.source_file = tempfile.NamedTemporaryFile()
    self.target_file = tempfile.NamedTemporaryFile()
    self.source_file.write("\n".join(self.source_lines).encode("utf-8"))
    self.source_file.flush()
    self.target_file.write("\n".join(self.target_lines).encode("utf-8"))
    self.target_file.flush()

  def tearDown(self):
    super(ParallelDataProviderTest, self).tearDown()
    self.source_file.close()
    self.target_file.close()

  def test_reading(self):
    num_epochs = 50
    data_provider = make_parallel_data_provider(
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
      item_dict["target_tokens"] = np.char.decode(
          item_dict["target_tokens"].astype("S"), "utf-8")
      item_dict["source_tokens"] = np.char.decode(
          item_dict["source_tokens"].astype("S"), "utf-8")

      # Source is Data + SEQUENCE_END
      self.assertEqual(item_dict["source_len"], 2)
      self.assertEqual(item_dict["source_tokens"][-1], "SEQUENCE_END")
      # Target is SEQUENCE_START + Data + SEQUENCE_END
      self.assertEqual(item_dict["target_len"], 3)
      self.assertEqual(item_dict["target_tokens"][0], "SEQUENCE_START")
      self.assertEqual(item_dict["target_tokens"][-1], "SEQUENCE_END")

      # Make sure data is aligned
      source_joined = " ".join(item_dict["source_tokens"][:-1])
      expected_target = self.source_to_target[source_joined]
      np.testing.assert_array_equal(
          item_dict["target_tokens"],
          ["SEQUENCE_START"] + expected_target.split(" ") + ["SEQUENCE_END"])

  def test_reading_without_targets(self):
    num_epochs = 50
    data_provider = make_parallel_data_provider(
        data_sources_source=[self.source_file.name],
        data_sources_target=None,
        num_epochs=num_epochs,
        shuffle=True)

    item_keys = list(data_provider.list_items())
    item_values = data_provider.get(item_keys)
    items_dict = dict(zip(item_keys, item_values))

    self.assertEqual(set(item_keys), set(["source_tokens", "source_len"]))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        item_dicts_ = [sess.run(items_dict) for _ in range(num_epochs * 3)]

    for item_dict in item_dicts_:
      self.assertEqual(item_dict["source_len"], 2)
      item_dict["source_tokens"] = np.char.decode(
          item_dict["source_tokens"].astype("S"), "utf-8")
      self.assertEqual(item_dict["source_tokens"][-1], "SEQUENCE_END")


if __name__ == "__main__":
  tf.test.main()
