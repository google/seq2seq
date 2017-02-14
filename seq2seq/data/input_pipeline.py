"""
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow as tf

from seq2seq.data import split_tokens_decoder, parallel_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

@six.add_metaclass(abc.ABCMeta)
class InputPipeline(object):
  def __init__(self, shuffle, num_epochs):
    self._shuffle = shuffle
    self._num_epochs = num_epochs

  def make_data_provider(self, **kwargs):
    raise NotImplementedError("Not implemented.")

  @property
  def feature_keys(self):
    return set()

  @property
  def label_keys(self):
    return set()

  @staticmethod
  def read_from_data_provider(data_provider):
    item_values = data_provider.get(list(data_provider.list_items()))
    items_dict = dict(zip(data_provider.list_items(), item_values))
    return items_dict


class TFRecordInputPipeline(InputPipeline):
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



class ParallelTextInputPipeline(InputPipeline):
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


