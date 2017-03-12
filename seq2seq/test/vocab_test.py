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

import tensorflow as tf
import numpy as np

from seq2seq.data import vocab
from seq2seq.test import utils as test_utils


class VocabInfoTest(tf.test.TestCase):
  """Tests VocabInfo class"""

  def setUp(self):
    super(VocabInfoTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.vocab_list = ["Hello", ".", "Bye"]
    self.vocab_file = test_utils.create_temporary_vocab_file(self.vocab_list)

  def tearDown(self):
    super(VocabInfoTest, self).tearDown()
    self.vocab_file.close()

  def test_vocab_info(self):
    vocab_info = vocab.get_vocab_info(self.vocab_file.name)
    self.assertEqual(vocab_info.vocab_size, 3)
    self.assertEqual(vocab_info.path, self.vocab_file.name)
    self.assertEqual(vocab_info.special_vocab.UNK, 3)
    self.assertEqual(vocab_info.special_vocab.SEQUENCE_START, 4)
    self.assertEqual(vocab_info.special_vocab.SEQUENCE_END, 5)
    self.assertEqual(vocab_info.total_size, 6)


class CreateVocabularyLookupTableTest(tf.test.TestCase):
  """
  Tests Vocabulary lookup table operations.
  """

  def test_without_counts(self):
    vocab_list = ["Hello", ".", "笑"]
    vocab_file = test_utils.create_temporary_vocab_file(vocab_list)

    vocab_to_id_table, id_to_vocab_table, _, vocab_size = \
      vocab.create_vocabulary_lookup_table(vocab_file.name)

    self.assertEqual(vocab_size, 6)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      ids = vocab_to_id_table.lookup(
          tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      ids = sess.run(ids)
      np.testing.assert_array_equal(ids, [0, 1, 2, 3, 3])

      words = id_to_vocab_table.lookup(
          tf.convert_to_tensor(
              [0, 1, 2, 3], dtype=tf.int64))
      words = sess.run(words)
      np.testing.assert_array_equal(
          np.char.decode(words.astype("S"), "utf-8"),
          ["Hello", ".", "笑", "UNK"])

  def test_with_counts(self):
    vocab_list = ["Hello", ".", "笑"]
    vocab_counts = [100, 200, 300]
    vocab_file = test_utils.create_temporary_vocab_file(vocab_list,
                                                        vocab_counts)

    vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size = \
      vocab.create_vocabulary_lookup_table(vocab_file.name)

    self.assertEqual(vocab_size, 6)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      ids = vocab_to_id_table.lookup(
          tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      ids = sess.run(ids)
      np.testing.assert_array_equal(ids, [0, 1, 2, 3, 3])

      words = id_to_vocab_table.lookup(
          tf.convert_to_tensor(
              [0, 1, 2, 3], dtype=tf.int64))
      words = sess.run(words)
      np.testing.assert_array_equal(
          np.char.decode(words.astype("S"), "utf-8"),
          ["Hello", ".", "笑", "UNK"])

      counts = word_to_count_table.lookup(
          tf.convert_to_tensor(["Hello", ".", "笑", "??", "xxx"]))
      counts = sess.run(counts)
      np.testing.assert_array_equal(counts, [100, 200, 300, -1, -1])


if __name__ == "__main__":
  tf.test.main()
