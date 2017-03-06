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
Abstract base class for tasks supported by the framework.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc

import six

import tensorflow as tf

from seq2seq.training import hooks
from seq2seq.data import vocab
from seq2seq import models
from seq2seq.configurable import Configurable
from seq2seq.metrics.metric_specs import METRIC_SPECS_DICT

@six.add_metaclass(abc.ABCMeta)
class Task(Configurable):
  def __init__(self, params):
    super(Task, self).__init__(params, None)
    self._model_cls = locate(self.params["model_class"]) or \
      getattr(models, self.params["model_class"])

  @staticmethod
  def default_params():
    return {
        "metrics": [],
        "model_class": "",
        "model_params": {},
    }

  @abc.abstractmethod
  def create_model(self, mode):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_training_hooks(self, estimator):
    output_dir = estimator.model_dir
    training_hooks = []
    model_analysis_hook = hooks.PrintModelAnalysisHook(
        filename=os.path.join(output_dir, "model_analysis.txt"))
    training_hooks.append(model_analysis_hook)

    metadata_hook = hooks.MetadataCaptureHook(
        output_dir=os.path.join(output_dir, "metadata"),
        step=10)
    training_hooks.append(metadata_hook)
    return training_hooks

  def create_metrics(self):
    return {m : METRIC_SPECS_DICT[m] for m in self.params["metrics"]}


class TextToText(Task):
  def __init__(self, params, mode):
    super(TextToText, self).__init__(params, mode)
    # Load vocabulary info
    self._source_vocab_info = vocab.get_vocab_info(self.params["vocab_source"])
    self._target_vocab_info = vocab.get_vocab_info(self.params["vocab_target"])

  @staticmethod
  def default_params():
    params = Task.default_params()
    params.update({
        "delimiter_source": " ",
        "delimiter_target": " ",
        "metrics": ["bleu", "log_perplexity",
                    "rouge_1/f_score", "rouge_1/r_score", "rouge_1/p_score",
                    "rouge_2/f_score", "rouge_2/r_score", "rouge_2/p_score",
                    "rouge_l/f_score"],
        "train_sample_frequency": 1000,
        "vocab_source": "",
        "vocab_target": ""
    })
    return params

  def create_model(self, mode):
    self._model_cls(
        source_vocab_info=self._source_vocab_info,
        target_vocab_info=self._target_vocab_info,
        params=self.params["model_params"],
        mode=mode)

  def create_training_hooks(self, estimator):
    output_dir = estimator.model_dir
    training_hooks = super(TextToText, self).create_training_hooks(estimator)

    train_sample_hook = hooks.TrainSampleHook(
        every_n_steps=self.params["train_sample_frequency"],
        sample_dir=os.path.join(output_dir, "samples"),
        source_delimiter=self.params["source_delimiter"],
        target_delimiter=self.params["target_delimiter"])
    training_hooks.append(train_sample_hook)

    tokens_per_sec_counter = hooks.TokensPerSecondCounter(
        every_n_steps=100,
        output_dir=output_dir)
    training_hooks.append(tokens_per_sec_counter)

    return training_hooks
