"""Collection of utility functions to read data
"""

import tensorflow as tf

from seq2seq.data import split_tokens_decoder, parallel_data_provider

def make_parallel_data_provider(data_sources_source, data_sources_target,
                                reader=tf.TextLineReader,
                                num_samples=None, **kwargs):
  """Creates a DataProvider that reads parallel text data.

  Args:
    data_sources_source: A list of data sources for the source text files.
    data_sources_target: A list of data sources for the target text files.
    num_samples: Optional, number of records in the dataset
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A DataProvider instance
  """

  # Decoders for both data sources
  decoder_source = split_tokens_decoder.SplitTokensDecoder(
      tokens_feature_name="source_tokens",
      length_feature_name="source_len")
  decoder_target = split_tokens_decoder.SplitTokensDecoder(
      tokens_feature_name="target_tokens",
      length_feature_name="target_len")

  # Datasets for both data sources
  dataset_source = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_source,
      reader=reader,
      decoder=decoder_source,
      num_samples=num_samples,
      items_to_descriptions={})
  dataset_target = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources_target,
      reader=reader,
      decoder=decoder_target,
      num_samples=num_samples,
      items_to_descriptions={})

  return parallel_data_provider.ParallelDataProvider(
      dataset1=dataset_source,
      dataset2=dataset_target,
      **kwargs)


def make_tfrecord_data_provider(data_sources, reader=tf.TFRecordReader,
                                num_samples=None, **kwargs):
  """
  Creates a TF Slim DatasetDataProvider for a list of input files.

  Args:
    data_sources: A list of input paths
    num_samples: Optional, number of records in the dataset
    kwargs: Additional arguments (shuffle, num_epochs, etc) that are passed
      to the data provider

  Returns:
    A DataProvider instance
  """

  keys_to_features = {
      "pair_id": tf.FixedLenFeature(
          [], dtype=tf.string),
      "source_len": tf.FixedLenFeature(
          [], dtype=tf.int64),
      "target_len": tf.FixedLenFeature(
          [], dtype=tf.int64),
      "source_tokens": tf.VarLenFeature(tf.string),
      "target_tokens": tf.VarLenFeature(tf.string)
  }

  items_to_handlers = {
      "pair_id": tf.contrib.slim.tfexample_decoder.Tensor("pair_id"),
      "source_len": tf.contrib.slim.tfexample_decoder.Tensor("source_len"),
      "target_len": tf.contrib.slim.tfexample_decoder.Tensor("target_len"),
      "source_tokens": tf.contrib.slim.tfexample_decoder.Tensor(
          "source_tokens", default_value=""),
      "target_tokens": tf.contrib.slim.tfexample_decoder.Tensor(
          "target_tokens", default_value="")
  }

  decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  dataset = tf.contrib.slim.dataset.Dataset(
      data_sources=data_sources,
      reader=reader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions={})

  return tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                   **kwargs)


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
