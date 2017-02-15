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
Unit tests for attention functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from seq2seq.decoders.attention import AttentionLayer


class AttentionLayerTest(tf.test.TestCase):
  """
  Tests the AttentionLayer module.
  """

  def setUp(self):
    super(AttentionLayerTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 8
    self.attention_dim = 128
    self.input_dim = 16
    self.seq_len = 10
    self.state_dim = 32

  def _test_with_score_type(self, score_type):
    """Tests Attention layer with a  given score type"""
    inputs_pl = tf.placeholder(tf.float32, (None, None, self.input_dim))
    state_pl = tf.placeholder(tf.float32, (None, self.state_dim))
    attention_fn = AttentionLayer(self.attention_dim, score_type=score_type)
    scores, context = attention_fn(state_pl, inputs_pl)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {}
      feed_dict[inputs_pl] = np.random.randn(self.batch_size, self.seq_len,
                                             self.input_dim)
      feed_dict[state_pl] = np.random.randn(self.batch_size, self.state_dim)
      scores_, context_ = sess.run([scores, context], feed_dict)

    np.testing.assert_array_equal(scores_.shape,
                                  [self.batch_size, self.seq_len])
    np.testing.assert_array_equal(context_.shape,
                                  [self.batch_size, self.input_dim])

    # Scores should sum to 1
    scores_sum = np.sum(scores_, axis=1)
    np.testing.assert_array_almost_equal(scores_sum, np.ones([self.batch_size]))

  def test_bahdanau(self):
    return self._test_with_score_type("bahdanau")

  def test_dot(self):
    return self._test_with_score_type("dot")


if __name__ == "__main__":
  tf.test.main()
