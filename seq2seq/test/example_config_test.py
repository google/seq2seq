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
Test Cases for example configuration files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pydoc import locate

import yaml

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.test.models_test import EncoderDecoderTests
from seq2seq import models

EXAMPLE_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../example_configs"))

def _load_model_from_config(config_path, hparam_overrides, vocab_info, mode):
  """Loads model from a configuration file"""
  with gfile.GFile(config_path) as config_file:
    config = yaml.load(config_file)
  model_cls = locate(config["model"]) or getattr(models, config["model"])
  hparams = config["hparams"]
  if hparam_overrides:
    hparams.update(hparam_overrides)
  # Change the max decode length to make the test run faster
  hparams["inference.max_decode_length"] = 5
  return model_cls(
      source_vocab_info=vocab_info,
      target_vocab_info=vocab_info,
      params=hparams,
      mode=mode)

# We only want to test the configuration - these tests are
# irrelevant for that
delattr(EncoderDecoderTests, "test_pipeline_train")
delattr(EncoderDecoderTests, "test_pipeline_inference")
delattr(EncoderDecoderTests, "test_pipeline_beam_search_infer")

class ExampleConfigTest(object):
  """Interface for configuration-based tests"""
  def __init__(self, *args, **kwargs):
    super(ExampleConfigTest, self).__init__(*args, **kwargs)
    self.vocab_info = None

  def _config_path(self):
    """Returns the path to the configuration to be tested"""
    raise NotImplementedError()

  def create_model(self, mode, params=None):
    """Creates the model"""
    return _load_model_from_config(
        config_path=self._config_path(),
        hparam_overrides=params,
        vocab_info=self.vocab_info,
        mode=mode)

class TestNMTLarge(ExampleConfigTest, EncoderDecoderTests):
  """Tests nmt_large.yml"""
  def _config_path(self):
    return os.path.join(EXAMPLE_CONFIG_DIR, "nmt_large.yml")

class TestNMTMedium(ExampleConfigTest, EncoderDecoderTests):
  """Tests nmt_medium.yml"""
  def _config_path(self):
    return os.path.join(EXAMPLE_CONFIG_DIR, "nmt_medium.yml")

class TestNMTSmall(ExampleConfigTest, EncoderDecoderTests):
  """Tests nmt_small.yml"""
  def _config_path(self):
    return os.path.join(EXAMPLE_CONFIG_DIR, "nmt_small.yml")

if __name__ == "__main__":
  tf.test.main()
