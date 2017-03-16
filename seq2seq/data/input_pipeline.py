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
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys

import six

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from seq2seq.configurable import Configurable
from seq2seq.data import split_tokens_decoder, parallel_data_provider
from seq2seq.data.sequence_example_decoder import TFSEquenceExampleDecoder


def make_input_pipeline_from_def(def_dict, mode, **kwargs):
  """Creates an InputPipeline object from a dictionary definition.

  Args:
    def_dict: A dictionary defining the input pipeline.
      It must have "class" and "params" that correspond to the class
      name and constructor parameters of an InputPipeline, respectively.
    mode: A value in tf.contrib.learn.ModeKeys

  Returns:
    A new InputPipeline object
  """
  if not "class" in def_dict:
    raise ValueError("Input Pipeline definition must have a class property.")

  class_ = def_dict["class"]
  if not hasattr(sys.modules[__name__], class_):
    raise ValueError("Invalid Input Pipeline class: {}".format(class_))

  pipeline_class = getattr(sys.modules[__name__], class_)

  # Constructor arguments
  params = {}
  if "params" in def_dict:
    params.update(def_dict["params"])
  params.update(kwargs)

  return pipeline_class(params=params, mode=mode)


@six.add_metaclass(abc.ABCMeta)
class InputPipeline(Configurable):
  """Abstract InputPipeline class. All input pipelines must inherit from this.
  An InputPipeline defines how data is read, parsed, and separated into
  features and labels.

  Params:
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """

  def __init__(self, params, mode):
    Configurable.__init__(self, params, mode)

  @staticmethod
  def default_params():
    return {
        "shuffle": True,
        "num_epochs": None,
    }

  def make_data_provider(self, **kwargs):
    """Creates DataProvider instance for this input pipeline. Additional
    keyword arguments are passed to the DataProvider.
    """
    raise NotImplementedError("Not implemented.")

  @property
  def feature_keys(self):
    """Defines the features that this input pipeline provides. Returns
      a set of strings.
    """
    return set()

  @property
  def label_keys(self):
    """Defines the labels that this input pipeline provides. Returns
      a set of strings.
    """
    return set()

  @staticmethod
  def read_from_data_provider(data_provider):
    """Utility function to read all available items from a DataProvider.
    """
    item_values = data_provider.get(list(data_provider.list_items()))
    items_dict = dict(zip(data_provider.list_items(), item_values))
    return items_dict


class ParallelTextInputPipeline(InputPipeline):
  """An input pipeline that reads two parallel (line-by-line aligned) text
  files.

  Params:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These must
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "source_files": [],
        "target_files": [],
        "source_delimiter": " ",
        "target_delimiter": " ",
    })
    return params

  def make_data_provider(self, **kwargs):
    decoder_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    dataset_source = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["source_files"],
        reader=tf.TextLineReader,
        decoder=decoder_source,
        num_samples=None,
        items_to_descriptions={})

    dataset_target = None
    if len(self.params["target_files"]) > 0:
      decoder_target = split_tokens_decoder.SplitTokensDecoder(
          tokens_feature_name="target_tokens",
          length_feature_name="target_len",
          prepend_token="SEQUENCE_START",
          append_token="SEQUENCE_END",
          delimiter=self.params["target_delimiter"])

      dataset_target = tf.contrib.slim.dataset.Dataset(
          data_sources=self.params["target_files"],
          reader=tf.TextLineReader,
          decoder=decoder_target,
          num_samples=None,
          items_to_descriptions={})

    return parallel_data_provider.ParallelDataProvider(
        dataset1=dataset_source,
        dataset2=dataset_target,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class TFRecordInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source
  and target sequences.

  Params:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "source_field": "source",
        "target_field": "target",
        "source_delimiter": " ",
        "target_delimiter": " ",
    })
    return params

  def make_data_provider(self, **kwargs):

    splitter_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self.params["source_delimiter"])

    splitter_target = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=self.params["target_delimiter"])

    keys_to_features = {
        self.params["source_field"]: tf.FixedLenFeature((), tf.string),
        self.params["target_field"]: tf.FixedLenFeature(
            (), tf.string, default_value="")
    }

    items_to_handlers = {}
    items_to_handlers["source_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(
            dict[self.params["source_field"]], ["source_tokens"])[0])
    items_to_handlers["source_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["source_field"]],
        func=lambda dict: splitter_source.decode(
            dict[self.params["source_field"]], ["source_len"])[0])
    items_to_handlers["target_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(
            dict[self.params["target_field"]], ["target_tokens"])[0])
    items_to_handlers["target_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self.params["target_field"]],
        func=lambda dict: splitter_target.decode(
            dict[self.params["target_field"]], ["target_len"])[0])

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class ImageCaptioningInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source
  and target sequences.

  Params:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
  """

  @staticmethod
  def default_params():
    params = InputPipeline.default_params()
    params.update({
        "files": [],
        "image_field": "image/data",
        "image_format": "jpg",
        "caption_ids_field": "image/caption_ids",
        "caption_tokens_field": "image/caption",
    })
    return params

  def make_data_provider(self, **kwargs):

    context_keys_to_features = {
        self.params["image_field"]: tf.FixedLenFeature(
            [], dtype=tf.string),
        "image/format": tf.FixedLenFeature(
            [], dtype=tf.string, default_value=self.params["image_format"]),
    }

    sequence_keys_to_features = {
        self.params["caption_ids_field"]: tf.FixedLenSequenceFeature(
            [], dtype=tf.int64),
        self.params["caption_tokens_field"]: tf.FixedLenSequenceFeature(
            [], dtype=tf.string)
    }

    items_to_handlers = {
        "image": tfexample_decoder.Image(
            image_key=self.params["image_field"],
            format_key="image/format",
            channels=3),
        "target_ids":
        tfexample_decoder.Tensor(self.params["caption_ids_field"]),
        "target_tokens":
        tfexample_decoder.Tensor(self.params["caption_tokens_field"]),
        "target_len": tfexample_decoder.ItemHandlerCallback(
            keys=[self.params["caption_tokens_field"]],
            func=lambda x: tf.size(x[self.params["caption_tokens_field"]]))
    }

    decoder = TFSEquenceExampleDecoder(
        context_keys_to_features, sequence_keys_to_features, items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self.params["files"],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset,
        shuffle=self.params["shuffle"],
        num_epochs=self.params["num_epochs"],
        **kwargs)

  @property
  def feature_keys(self):
    return set(["image"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_ids", "target_len"])
