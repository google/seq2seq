"""Collection of utility functions to read data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq.data import split_tokens_decoder, parallel_data_provider


def make_parallel_data_provider(data_sources_source,
                                data_sources_target,
                                reader=tf.TextLineReader,
                                num_samples=None,
                                delimiter=" ",
                                **kwargs):
  """Creates a DataProvider that reads parallel text data.

  Args:
    data_sources_source: A list of data sources for the source text files.
    data_sources_target: A list of data sources for the target text files.
      Can be None for inference mode.
    num_samples: Optional, number of records in the dataset
    delimiter: Split tokens in the data on this delimiter. Defaults to space.
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A DataProvider instance
  """

  decoder_source = split_tokens_decoder.SplitTokensDecoder(
      tokens_feature_name="source_tokens",
      length_feature_name="source_len",
      append_token="SEQUENCE_END",
      delimiter=delimiter)

  dataset_source = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_source,
      reader=reader,
      decoder=decoder_source,
      num_samples=num_samples,
      items_to_descriptions={})

  dataset_target = None
  if data_sources_target is not None:
    decoder_target = split_tokens_decoder.SplitTokensDecoder(
        tokens_feature_name="target_tokens",
        length_feature_name="target_len",
        prepend_token="SEQUENCE_START",
        append_token="SEQUENCE_END",
        delimiter=delimiter)

    dataset_target = tf.contrib.slim.dataset.Dataset(
        data_sources=data_sources_target,
        reader=reader,
        decoder=decoder_target,
        num_samples=num_samples,
        items_to_descriptions={})

  return parallel_data_provider.ParallelDataProvider(
      dataset1=dataset_source, dataset2=dataset_target, **kwargs)


def read_from_data_provider(data_provider):
  """Reads all fields from a data provider.

  Args:
    data_provider: A DataProvider instance

  Returns:
    A dictionary of tensors corresponding to all features
    defined by the DataProvider
  """
  item_values = data_provider.get(list(data_provider.list_items()))
  items_dict = dict(zip(data_provider.list_items(), item_values))
  return items_dict
