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
Test Cases for Inference utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import tensorflow as tf
import numpy as np

from seq2seq.inference import inference

class TestUnkMapping(tf.test.TestCase):
  """Tests inference.get_unk_mapping"""

  def test_read_mapping(self):
    mapping_file = tempfile.NamedTemporaryFile()
    mapping_file.write("a\tb\n".encode("utf-8"))
    mapping_file.write("b\tc\n".encode("utf-8"))
    mapping_file.write("c\td\n".encode("utf-8"))
    mapping_file.flush()

    mapping_dict = inference.get_unk_mapping(mapping_file.name)
    self.assertEqual(
        mapping_dict,
        {"a": "b", "b": "c", "c": "d"})

    mapping_file.close()


class TestUnkReplace(tf.test.TestCase):
  """Tests inference.unk_replace"""

  def test_without_mapping(self):
    #pylint: disable=no-self-use
    source_tokens = "A B C D".split(" ")
    predicted_tokens = "1 2 UNK 4".split(" ")
    attention_scores = np.identity(4)

    new_tokens = inference.unk_replace(
        source_tokens=source_tokens,
        predicted_tokens=predicted_tokens,
        attention_scores=attention_scores)

    np.testing.assert_array_equal(new_tokens, "1 2 C 4".split(" "))

  def test_with_mapping(self):
    #pylint: disable=no-self-use
    source_tokens = "A B C D A".split(" ")
    predicted_tokens = "1 2 UNK 4 UNK".split(" ")
    attention_scores = np.identity(5)
    mapping = {"C": "3"}

    new_tokens = inference.unk_replace(
        source_tokens=source_tokens,
        predicted_tokens=predicted_tokens,
        attention_scores=attention_scores,
        mapping=mapping)

    np.testing.assert_array_equal(new_tokens, "1 2 3 4 A".split(" "))

if __name__ == '__main__':
  tf.test.main()
