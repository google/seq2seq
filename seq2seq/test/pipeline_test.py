# -*- coding: utf-8 -*-

"""
Test Cases for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import imp
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

from seq2seq.test import utils as test_utils

BIN_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../bin"))

def _clear_flags():
  """Resets Tensorflow's FLAG values"""
  #pylint: disable=W0212
  tf.app.flags.FLAGS = tf.python.platform.flags._FlagValues()
  tf.app.flags._global_parser = argparse.ArgumentParser()

class TrainTest(tf.test.TestCase):
  """Tests training and inference scripts.
  """
  def setUp(self):
    super(TrainTest, self).setUp()
    self.output_dir = tempfile.mkdtemp()
    self.bin_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../bin"))

  def tearDown(self):
    shutil.rmtree(self.output_dir, ignore_errors=True)
    super(TrainTest, self).tearDown()

  def test_train_infer(self):
    """Tests training and inference scripts.
    """
    # Create dummy data
    sources_train, targets_train = test_utils.create_temp_parallel_data(
        sources=["a a a a", "b b b b", "c c c c", "笑 笑 笑 笑"],
        targets=["b b b b", "a a a a", "c c c c", "泣 泣 泣 泣"])
    sources_dev, targets_dev = test_utils.create_temp_parallel_data(
        sources=["a a", "b b", "c c c", "笑 笑 笑"],
        targets=["b b", "a a", "c c c", "泣 泣 泣"])
    vocab_source = test_utils.create_temporary_vocab_file(["a", "b", "c", "笑"])
    vocab_target = test_utils.create_temporary_vocab_file(["a", "b", "c", "泣"])

    _clear_flags()
    tf.reset_default_graph()
    train_script = imp.load_source(
        "seq2seq.test.train_bin", os.path.join(BIN_FOLDER, "train.py"))

    # Set training flags
    tf.app.flags.FLAGS.output_dir = self.output_dir
    tf.app.flags.FLAGS.train_source = sources_train.name
    tf.app.flags.FLAGS.train_target = targets_train.name
    tf.app.flags.FLAGS.dev_source = sources_dev.name
    tf.app.flags.FLAGS.dev_target = targets_dev.name
    tf.app.flags.FLAGS.vocab_source = vocab_source.name
    tf.app.flags.FLAGS.vocab_target = vocab_target.name
    tf.app.flags.FLAGS.model = "AttentionSeq2Seq"
    tf.app.flags.FLAGS.batch_size = 2
    tf.app.flags.FLAGS.train_steps = 50

    # Run training
    tf.logging.set_verbosity(tf.logging.INFO)
    train_script.main([])

    # Make sure a checkpoint was written
    expected_checkpoint = os.path.join(
        self.output_dir, "model.ckpt-50.data-00000-of-00001")
    self.assertTrue(os.path.exists(expected_checkpoint))

    # Reset flags and import inference script
    _clear_flags()
    tf.reset_default_graph()
    infer_script = imp.load_source(
        "seq2seq.test.infer_bin", os.path.join(BIN_FOLDER, "infer.py"))

    # Set inference flags
    tf.app.flags.FLAGS.model_dir = self.output_dir
    tf.app.flags.FLAGS.source = sources_dev.name
    tf.app.flags.FLAGS.vocab_source = vocab_source.name
    tf.app.flags.FLAGS.vocab_target = vocab_target.name
    tf.app.flags.FLAGS.model = "AttentionSeq2Seq"
    tf.app.flags.FLAGS.batch_size = 2
    tf.app.flags.FLAGS.checkpoint_path = os.path.join(
        self.output_dir, "model.ckpt-50")

    # Make sure inference runs successfully
    infer_script.main([])

    # Visualize attention scores
    _clear_flags()
    tf.reset_default_graph()
    print_attention_script = imp.load_source(
        "seq2seq.test.print_attention_bin",
        os.path.join(BIN_FOLDER, "print_attention.py"))

    attention_dir = os.path.join(self.output_dir, "att")

    # Set flags
    tf.app.flags.FLAGS.output_dir = attention_dir
    tf.app.flags.FLAGS.model_dir = self.output_dir
    tf.app.flags.FLAGS.source = sources_dev.name
    tf.app.flags.FLAGS.vocab_source = vocab_source.name
    tf.app.flags.FLAGS.vocab_target = vocab_target.name
    tf.app.flags.FLAGS.model = "AttentionSeq2Seq"
    tf.app.flags.FLAGS.batch_size = 2
    tf.app.flags.FLAGS.checkpoint_path = os.path.join(
        self.output_dir, "model.ckpt-50")

    # Run the print_attention script
    print_attention_script.main([])

    # Make sure scores and visualizations exist
    self.assertTrue(os.path.exists(os.path.join(
        attention_dir, "attention_scores.npz")))
    self.assertTrue(os.path.exists(os.path.join(
        attention_dir, "00002.png")))

    # Load attention scores and assert shape
    scores = np.load(os.path.join(attention_dir, "attention_scores.npz"))
    self.assertIn("arr_0", scores)
    self.assertEqual(scores["arr_0"].shape[1], 3)
    self.assertIn("arr_1", scores)
    self.assertEqual(scores["arr_1"].shape[1], 3)
    self.assertIn("arr_2", scores)
    self.assertEqual(scores["arr_2"].shape[1], 4)
    self.assertIn("arr_3", scores)
    self.assertEqual(scores["arr_3"].shape[1], 4)


if __name__ == "__main__":
  tf.test.main()
