"""Tests for SessionRunHooks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import shutil

import tensorflow as tf
from tensorflow.python.training import monitored_session
from tensorflow.python.platform import gfile

from seq2seq import graph_utils
from seq2seq.training import hooks
from seq2seq.test import utils as test_utils
from seq2seq.data import vocab

class TestPrintModelAnalysisHook(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""
  def test_begin(self):
    outfile = tempfile.NamedTemporaryFile()
    tf.get_variable("weigths", [128, 128])
    hook = hooks.PrintModelAnalysisHook(filename=outfile.name)
    hook.begin()
    file_contents = outfile.read().strip()
    self.assertEqual(
        file_contents.decode(),
        "_TFProfRoot (--/16.38k params)\n"
        "  weigths (128x128, 16.38k/16.38k params)")
    outfile.close()


class TestTrainSampleHook(tf.test.TestCase):
  """Tests `TrainSampleHook` class.
  """
  def setUp(self):
    super(TestTrainSampleHook, self).setUp()

    # The hook expects these collections to be in the graph
    graph_utils.add_dict_to_collection(
        {"predictions": tf.constant([[2, 3]], dtype=tf.int64)},
        "model_output")
    graph_utils.add_dict_to_collection(
        {"source_ids": tf.constant([[1, 2]])},
        "features")
    graph_utils.add_dict_to_collection({
        "target_tokens": tf.constant([["Hello", "World"]]),
        "target_len": tf.constant([2])},
      "labels")

    # Create vocabulary
    self.vocab_file = test_utils.create_temporary_vocab_file(
        ["Hello", "World", "!"])
    _, target_id_to_vocab, _ = vocab.create_vocabulary_lookup_table(
        self.vocab_file.name)
    tf.add_to_collection("target_id_to_vocab", target_id_to_vocab)

  def tearDown(self):
    super(TestTrainSampleHook, self).tearDown()
    self.vocab_file.close()

  def test_sampling(self):
    outfile = tempfile.NamedTemporaryFile()
    hook = hooks.TrainSampleHook(every_n_steps=10, file=outfile.name)

    global_step = tf.contrib.framework.get_or_create_global_step()
    no_op = tf.no_op()
    hook.begin()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.initialize_all_tables())

      #pylint: disable=W0212
      mon_sess = monitored_session._HookedSession(sess, [hook])
      # Should trigger for step 0
      sess.run(tf.assign(global_step, 0))
      mon_sess.run(no_op)
      self.assertIn("Prediction followed by Target @ Step 0",
                    outfile.read().decode())
      # Should not trigger for step 9
      sess.run(tf.assign(global_step, 9))
      mon_sess.run(no_op)
      self.assertNotIn("Prediction followed by Target @ Step 9",
                       outfile.read().decode())
      # Should trigger for step 10
      sess.run(tf.assign(global_step, 10))
      mon_sess.run(no_op)
      self.assertIn("Prediction followed by Target @ Step 10",
                    outfile.read().decode())


class TestMetadataCaptureHook(tf.test.TestCase):
  """Test for the MetadataCaptureHook"""

  def setUp(self):
    super(TestMetadataCaptureHook, self).setUp()
    self.capture_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(TestMetadataCaptureHook, self).tearDown()
    shutil.rmtree(self.capture_dir)

  def test_capture(self):
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Some test computation
    some_weights = tf.get_variable("weigths", [2, 128])
    computation = tf.nn.softmax(some_weights)

    hook = hooks.MetadataCaptureHook(step=5, output_dir=self.capture_dir)
    hook.begin()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      #pylint: disable=W0212
      mon_sess = monitored_session._HookedSession(sess, [hook])
      # Should not trigger for step 0
      sess.run(tf.assign(global_step, 0))
      mon_sess.run(computation)
      self.assertEqual(
          gfile.ListDirectory(self.capture_dir),
          [])
      # Should trigger *after* step 5
      sess.run(tf.assign(global_step, 5))
      mon_sess.run(computation)
      self.assertEqual(
          gfile.ListDirectory(self.capture_dir),
          [])
      mon_sess.run(computation)
      self.assertEqual(
          set(gfile.ListDirectory(self.capture_dir)),
          set(["run_meta", "tfprof_log", "timeline.json"]))


if __name__ == "__main__":
  tf.test.main()
