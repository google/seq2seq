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
Unit tests for loss-related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from seq2seq import losses as seq2seq_losses
import tensorflow as tf
import numpy as np


class CrossEntropySequenceLossTest(tf.test.TestCase):
  """
  Test for `sqe2seq.losses.sequence_mask`.
  """

  def setUp(self):
    super(CrossEntropySequenceLossTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 10
    self.vocab_size = 50

  def test_op(self):
    logits = np.random.randn(self.sequence_length, self.batch_size,
                             self.vocab_size)
    logits = logits.astype(np.float32)
    sequence_length = np.array([1, 2, 3, 4])
    targets = np.random.randint(0, self.vocab_size,
                                [self.sequence_length, self.batch_size])
    losses = seq2seq_losses.cross_entropy_sequence_loss(logits, targets,
                                                        sequence_length)

    with self.test_session() as sess:
      losses_ = sess.run(losses)

    # Make sure all losses not past the sequence length are > 0
    np.testing.assert_array_less(np.zeros_like(losses_[:1, 0]), losses_[:1, 0])
    np.testing.assert_array_less(np.zeros_like(losses_[:2, 1]), losses_[:2, 1])
    np.testing.assert_array_less(np.zeros_like(losses_[:3, 2]), losses_[:3, 2])

    # Make sure all losses past the sequence length are 0
    np.testing.assert_array_equal(losses_[1:, 0], np.zeros_like(losses_[1:, 0]))
    np.testing.assert_array_equal(losses_[2:, 1], np.zeros_like(losses_[2:, 1]))
    np.testing.assert_array_equal(losses_[3:, 2], np.zeros_like(losses_[3:, 2]))


if __name__ == "__main__":
  tf.test.main()
