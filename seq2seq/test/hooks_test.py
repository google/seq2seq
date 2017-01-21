# -*- coding: utf-8 -*-

"""Tests for SessionRunHooks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import shutil
import time

import tensorflow as tf
from tensorflow.python.training import monitored_session
from tensorflow.python.platform import gfile

from seq2seq import graph_utils
from seq2seq.training import hooks


class TestPrintModelAnalysisHook(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""

  def test_begin(self):
    outfile = tempfile.NamedTemporaryFile()
    tf.get_variable("weigths", [128, 128])
    hook = hooks.PrintModelAnalysisHook(filename=outfile.name)
    hook.begin()
    file_contents = outfile.read().strip()
    self.assertEqual(file_contents.decode(), "_TFProfRoot (--/16.38k params)\n"
                     "  weigths (128x128, 16.38k/16.38k params)")
    outfile.close()


class TestTrainSampleHook(tf.test.TestCase):
  """Tests `TrainSampleHook` class.
  """

  def setUp(self):
    super(TestTrainSampleHook, self).setUp()
    self.sample_dir = tempfile.mkdtemp()

    # The hook expects these collections to be in the graph
    pred_dict = {}
    pred_dict["predicted_tokens"] = tf.constant([["Hello", "World", "笑w"]])
    pred_dict["labels.target_tokens"] = tf.constant([["Hello", "World", "笑w"]])
    pred_dict["labels.target_len"] = tf.constant(2),
    graph_utils.add_dict_to_collection(pred_dict, "predictions")

  def tearDown(self):
    super(TestTrainSampleHook, self).tearDown()
    shutil.rmtree(self.sample_dir)

  def test_sampling(self):
    hook = hooks.TrainSampleHook(
        every_n_steps=10,
        sample_dir=self.sample_dir)

    global_step = tf.contrib.framework.get_or_create_global_step()
    no_op = tf.no_op()
    hook.begin()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      #pylint: disable=W0212
      mon_sess = monitored_session._HookedSession(sess, [hook])
      # Should trigger for step 0
      sess.run(tf.assign(global_step, 0))
      mon_sess.run(no_op)

      outfile = os.path.join(self.sample_dir, "samples_000000.txt")
      with open(outfile, "rb") as readfile:
        self.assertIn("Prediction followed by Target @ Step 0",
                      readfile.read().decode("utf-8"))

      # Should not trigger for step 9
      sess.run(tf.assign(global_step, 9))
      mon_sess.run(no_op)
      outfile = os.path.join(self.sample_dir, "samples_000009.txt")
      self.assertFalse(os.path.exists(outfile))

      # Should trigger for step 10
      sess.run(tf.assign(global_step, 10))
      mon_sess.run(no_op)
      outfile = os.path.join(self.sample_dir, "samples_000010.txt")
      with open(outfile, "rb") as readfile:
        self.assertIn("Prediction followed by Target @ Step 10",
                      readfile.read().decode("utf-8"))


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
      self.assertEqual(gfile.ListDirectory(self.capture_dir), [])
      # Should trigger *after* step 5
      sess.run(tf.assign(global_step, 5))
      mon_sess.run(computation)
      self.assertEqual(gfile.ListDirectory(self.capture_dir), [])
      mon_sess.run(computation)
      self.assertEqual(
          set(gfile.ListDirectory(self.capture_dir)),
          set(["run_meta", "tfprof_log", "timeline.json"]))


class TestTokenCounter(tf.test.TestCase):
  """Tests the TokensPerSecondCounter hook"""

  def setUp(self):
    super(TestTokenCounter, self).setUp()
    self.summary_dir = tempfile.mkdtemp()
    graph_utils.add_dict_to_collection({
        "source_len": tf.constant([[2, 3]])
    }, "features")
    graph_utils.add_dict_to_collection({
        "target_len": tf.constant([4, 6])
    }, "labels")

  def tearDown(self):
    super(TestTokenCounter, self).tearDown()
    shutil.rmtree(self.summary_dir, ignore_errors=True)

  def test_counter(self):
    graph = tf.get_default_graph()
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = tf.assign_add(global_step, 1)

    # Create the hook we want to test
    summary_writer = tf.contrib.testing.FakeSummaryWriter(self.summary_dir,
                                                          graph)
    hook = hooks.TokensPerSecondCounter(
        summary_writer=summary_writer, every_n_steps=10)
    hook.begin()

    # Run a few perations
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      #pylint: disable=W0212
      mon_sess = monitored_session._HookedSession(
          sess, [hook])
      for _ in range(30):
        time.sleep(0.01)
        mon_sess.run(train_op)
      hook.end(sess)

    summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.summary_dir,
        expected_graph=graph,
        expected_summaries={})
    # Hook should have triggered for global step 11 and 21
    self.assertItemsEqual([11, 21], summary_writer.summaries.keys())
    for step in [11, 21]:
      summary_value = summary_writer.summaries[step][0].value[0]
      self.assertEqual('tokens/sec', summary_value.tag)
      self.assertGreater(summary_value.simple_value, 0)


if __name__ == "__main__":
  tf.test.main()
