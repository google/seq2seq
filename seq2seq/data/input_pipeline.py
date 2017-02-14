"""
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import sys

import six
import yaml

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from seq2seq.data import split_tokens_decoder, parallel_data_provider

def make_input_pipeline_from_def(def_str, **kwargs):
  """Creates an InputPipeline object from a YAML or JSON definition string.

  Args:
    def_str: A YAML/JSON string that defines the input pipeline.
      It must have a "class" key and "args" they that correspond to the class
      name and constructor parameter of an InputPipeline, respectively.

  Returns:
    A new InputPipeline object
  """
  def_dict = yaml.load(def_str)

  if not "class" in def_dict:
    raise ValueError("Input Pipeline definition must have a class propert.")

  class_ = def_dict["class"]
  if not hasattr(sys.modules[__name__], class_):
    raise ValueError("Invalid Input Pipeline class: {}".format(class_))

  pipeline_class = getattr(sys.modules[__name__], class_)

  # Constructor arguments
  class_args = {}
  if "args" in def_dict:
    class_args.update(def_dict["args"])
  class_args.update(kwargs)

  return pipeline_class(**class_args)


@six.add_metaclass(abc.ABCMeta)
class InputPipeline(object):
  """Abstract InputPipeline class. All input pipelines must inherit from this.
  An InputPipeline defines how data is read, parsed, and separated into
  features and labels.

  Args:
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """
  def __init__(self, shuffle, num_epochs):
    self._shuffle = shuffle
    self._num_epochs = num_epochs

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

  Args:
    source_files: An array of file names for the source data.
    target_files: An array of file names for the target data. These mut
      be aligned to the `source_files`.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """
  def __init__(self, source_files, target_files,
               source_delimiter=" ", target_delimiter=" ",
               shuffle=False, num_epochs=None):
    super(ParallelTextInputPipeline, self).__init__(shuffle, num_epochs)
    self._source_files = source_files
    self._target_files = target_files
    self._source_delimiter = source_delimiter
    self._target_delimiter = target_delimiter

  def make_data_provider(self, **kwargs):
    decoder_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self._source_delimiter)

    dataset_source = tf.contrib.slim.dataset.Dataset(
        data_sources=self._source_files,
        reader=tf.TextLineReader,
        decoder=decoder_source,
        num_samples=None,
        items_to_descriptions={})

    dataset_target = None
    if self._target_files is not None:
      decoder_target = split_tokens_decoder.SplitTokensDecoder(
          tokens_feature_name="target_tokens",
          length_feature_name="target_len",
          prepend_token="SEQUENCE_START",
          append_token="SEQUENCE_END",
          delimiter=self._target_delimiter)

      dataset_target = tf.contrib.slim.dataset.Dataset(
          data_sources=self._target_files,
          reader=tf.TextLineReader,
          decoder=decoder_target,
          num_samples=None,
          items_to_descriptions={})

    return parallel_data_provider.ParallelDataProvider(
        dataset1=dataset_source, dataset2=dataset_target,
        shuffle=self._shuffle, num_epochs=self._num_epochs, **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])


class TFRecordInputPipeline(InputPipeline):
  """An input pipeline that reads a TFRecords containing both source
  and target sequences.

  Args:
    files: An array of file names to read from.
    source_field: The TFRecord feature field containing the source text.
    target_field: The TFRecord feature field containing the target text.
    source_delimiter: A character to split the source text on. Defaults
      to  " " (space). For character-level training this can be set to the
      empty string.
    target_delimiter: Same as `source_delimiter` but for the target text.
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """
  def __init__(self, files,
               source_field="source", target_field="target",
               source_delimiter=" ", target_delimiter=" ",
               shuffle=False, num_epochs=None):
    super(TFRecordInputPipeline, self).__init__(shuffle, num_epochs)
    self._files = files
    self._source_field = source_field
    self._target_field = target_field
    self._source_delimiter = source_delimiter
    self._target_delimiter = target_delimiter

  def make_data_provider(self, **kwargs):

    splitter_source = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="source_tokens",
        length_feature_name="source_len",
        append_token="SEQUENCE_END",
        delimiter=self._source_delimiter)

    splitter_target = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=self._target_delimiter)

    keys_to_features = {
        self._source_field: tf.FixedLenFeature((), tf.string),
        self._target_field: tf.FixedLenFeature((), tf.string, default_value="")
    }

    items_to_handlers = {}
    items_to_handlers["source_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self._source_field],
        func=lambda dict: splitter_source.decode(
            dict[self._source_field], ["source_tokens"])[0])
    items_to_handlers["source_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self._source_field],
        func=lambda dict: splitter_source.decode(
            dict[self._source_field], ["source_len"])[0])
    items_to_handlers["target_tokens"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self._target_field],
        func=lambda dict: splitter_target.decode(
            dict[self._target_field], ["target_tokens"])[0])
    items_to_handlers["target_len"] = tfexample_decoder.ItemHandlerCallback(
        keys=[self._target_field],
        func=lambda dict: splitter_target.decode(
            dict[self._target_field], ["target_len"])[0])

    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(
        data_sources=self._files,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={})

    return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset=dataset, shuffle=self._shuffle, num_epochs=self._num_epochs,
        **kwargs)

  @property
  def feature_keys(self):
    return set(["source_tokens", "source_len"])

  @property
  def label_keys(self):
    return set(["target_tokens", "target_len"])
