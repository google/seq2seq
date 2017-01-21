# -*- coding: utf-8 -*-

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

  def setUp(self):
    super(CreateVocabularyLookupTableTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.vocab_list = ["Hello", ".", "笑"]
    self.vocab_file = test_utils.create_temporary_vocab_file(self.vocab_list)

  def tearDown(self):
    super(CreateVocabularyLookupTableTest, self).tearDown()
    self.vocab_file.close()

  def test_lookup_table(self):

    vocab_to_id_table, id_to_vocab_table, vocab_size = \
      vocab.create_vocabulary_lookup_table(self.vocab_file.name)

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


if __name__ == "__main__":
  tf.test.main()
