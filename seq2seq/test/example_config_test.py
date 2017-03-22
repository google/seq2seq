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
from tensorflow import gfile

from seq2seq.test.models_test import EncoderDecoderTests
from seq2seq import models

EXAMPLE_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../example_configs"))


def _load_model_from_config(config_path, hparam_overrides, vocab_file, mode):
  """Loads model from a configuration file"""
  with gfile.GFile(config_path) as config_file:
    config = yaml.load(config_file)
  model_cls = locate(config["model"]) or getattr(models, config["model"])
  model_params = config["model_params"]
  if hparam_overrides:
    model_params.update(hparam_overrides)
  # Change the max decode length to make the test run faster
  model_params["decoder.params"]["max_decode_length"] = 5
  model_params["vocab_source"] = vocab_file
  model_params["vocab_target"] = vocab_file
  return model_cls(params=model_params, mode=mode)


class ExampleConfigTest(object):
  """Interface for configuration-based tests"""

  def __init__(self, *args, **kwargs):
    super(ExampleConfigTest, self).__init__(*args, **kwargs)
    self.vocab_file = None

  def _config_path(self):
    """Returns the path to the configuration to be tested"""
    raise NotImplementedError()

  def create_model(self, mode, params=None):
    """Creates the model"""
    return _load_model_from_config(
        config_path=self._config_path(),
        hparam_overrides=params,
        vocab_file=self.vocab_file.name,
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

class TestNMTConv(ExampleConfigTest, EncoderDecoderTests):
  """Tests nmt_small.yml"""

  def _config_path(self):
    return os.path.join(EXAMPLE_CONFIG_DIR, "nmt_conv.yml")


if __name__ == "__main__":
  tf.test.main()
