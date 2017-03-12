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
import yaml

from seq2seq.data import input_pipeline
from seq2seq.test import utils as test_utils


class TestInputPipelineDef(tf.test.TestCase):
  """Tests InputPipeline string definitions"""

  def test_without_extra_args(self):
    pipeline_def = yaml.load("""
      class: ParallelTextInputPipeline
      params:
        source_files: ["file1"]
        target_files: ["file2"]
        num_epochs: 1
        shuffle: True
    """)
    pipeline = input_pipeline.make_input_pipeline_from_def(
        pipeline_def, tf.contrib.learn.ModeKeys.TRAIN)
    self.assertIsInstance(pipeline, input_pipeline.ParallelTextInputPipeline)
    #pylint: disable=W0212
    self.assertEqual(pipeline.params["source_files"], ["file1"])
    self.assertEqual(pipeline.params["target_files"], ["file2"])
    self.assertEqual(pipeline.params["num_epochs"], 1)
    self.assertEqual(pipeline.params["shuffle"], True)

  def test_with_extra_args(self):
    pipeline_def = yaml.load("""
      class: ParallelTextInputPipeline
      params:
        source_files: ["file1"]
        target_files: ["file2"]
        num_epochs: 1
        shuffle: True
    """)
    pipeline = input_pipeline.make_input_pipeline_from_def(
        def_dict=pipeline_def,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        num_epochs=5,
        shuffle=False)
    self.assertIsInstance(pipeline, input_pipeline.ParallelTextInputPipeline)
    #pylint: disable=W0212
    self.assertEqual(pipeline.params["source_files"], ["file1"])
    self.assertEqual(pipeline.params["target_files"], ["file2"])
    self.assertEqual(pipeline.params["num_epochs"], 5)
    self.assertEqual(pipeline.params["shuffle"], False)


class TFRecordsInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(TFRecordsInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_pipeline(self):
    tfrecords_file = test_utils.create_temp_tfrecords(
        sources=["Hello World . 笑"], targets=["Bye 泣"])

    pipeline = input_pipeline.TFRecordInputPipeline(
        params={
            "files": [tfrecords_file.name],
            "source_field": "source",
            "target_field": "target",
            "num_epochs": 5,
            "shuffle": False
        },
        mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()

    features = pipeline.read_from_data_provider(data_provider)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        res = sess.run(features)

    self.assertEqual(res["source_len"], 5)
    self.assertEqual(res["target_len"], 4)
    np.testing.assert_array_equal(
        np.char.decode(res["source_tokens"].astype("S"), "utf-8"),
        ["Hello", "World", ".", "笑", "SEQUENCE_END"])
    np.testing.assert_array_equal(
        np.char.decode(res["target_tokens"].astype("S"), "utf-8"),
        ["SEQUENCE_START", "Bye", "泣", "SEQUENCE_END"])


class ParallelTextInputPipelineTest(tf.test.TestCase):
  """
  Tests Data Provider operations.
  """

  def setUp(self):
    super(ParallelTextInputPipelineTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)

  def test_pipeline(self):
    file_source, file_target = test_utils.create_temp_parallel_data(
        sources=["Hello World . 笑"], targets=["Bye 泣"])

    pipeline = input_pipeline.ParallelTextInputPipeline(
        params={
            "source_files": [file_source.name],
            "target_files": [file_target.name],
            "num_epochs": 5,
            "shuffle": False
        },
        mode=tf.contrib.learn.ModeKeys.TRAIN)

    data_provider = pipeline.make_data_provider()

    features = pipeline.read_from_data_provider(data_provider)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      with tf.contrib.slim.queues.QueueRunners(sess):
        res = sess.run(features)

    self.assertEqual(res["source_len"], 5)
    self.assertEqual(res["target_len"], 4)
    np.testing.assert_array_equal(
        np.char.decode(res["source_tokens"].astype("S"), "utf-8"),
        ["Hello", "World", ".", "笑", "SEQUENCE_END"])
    np.testing.assert_array_equal(
        np.char.decode(res["target_tokens"].astype("S"), "utf-8"),
        ["SEQUENCE_START", "Bye", "泣", "SEQUENCE_END"])


if __name__ == "__main__":
  tf.test.main()
