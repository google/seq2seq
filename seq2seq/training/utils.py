"""Miscellaneous training utility functions.
"""

from seq2seq.inputs import read_from_data_provider
import tensorflow as tf


def get_rnn_cell(cell_type,
                 num_units,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0):
  """Creates a new RNN Cell.

  Args:
    cell_type: A cell lass name defined in `tf.nn.rnn_cell`,
      e.g. `LSTMCell` or `GRUCell`
    num_units: Number of cell units
    num_layers: Number of layers. The cell will be wrapped with
      `tf.nn.rnn_cell.MultiRNNCell`
    dropout_input_keep_prob: Dropout keep probability applied
      to the input of cell *at each layer*
    dropout_output_keep_prob: Dropout keep probability applied
      to the output of cell *at each layer*

  Returns:
    An instance of `tf.nn.rnn_cell.RNNCell`.
  """
  #pylint: disable=redefined-variable-type
  cell_class = getattr(tf.nn.rnn_cell, cell_type)
  cell = cell_class(num_units)

  if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=cell,
        input_keep_prob=dropout_input_keep_prob,
        output_keep_prob=dropout_output_keep_prob)

  if num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

  return cell


def create_input_fn(data_provider_fn,
                    featurizer_fn,
                    batch_size,
                    bucket_boundaries=None):
  """Creates an input function that can be used with tf.learn estimators.
    Note that you must pass "factory funcitons" for both the data provider and
    featurizer to ensure that everything will be created in  the same graph.

  Args:
    data_provider_fn: Function that creates a data provider to read from.
      An instance of `tf.contrib.slim.data_provider.DataProvider`.
    featurizer_fn: A function that creates a featurizer function
      which takes tensors returned by the data provider and transfroms them
      into a (features, labels) tuple.
    batch_size: Create batches of this size. A queue to hold a
      reasonable number of batches in memory is created.
    bucket_boundaries: int list, increasing non-negative numbers.
      If None, no bucket is performed.

  Returns:
    An input function that returns `(feature_batch, labels_batch)`
    tuples when called.
  """

  def input_fn():
    """Creates features and labels.
    """
    features = read_from_data_provider(data_provider_fn())
    features, labels = featurizer_fn(features)

    # We need to merge features and labels so we can batch them together.
    feature_keys = features.keys()
    label_keys = labels.keys()
    features_and_labels = features.copy()
    features_and_labels.update(labels)

    if bucket_boundaries:
      bucket_num, batch = tf.contrib.training.bucket_by_sequence_length(
          input_length=features_and_labels["source_len"],
          bucket_boundaries=bucket_boundaries,
          tensors=features_and_labels,
          batch_size=batch_size,
          keep_input=features_and_labels["target_len"] >= 1,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          name="bucket_queue")
      tf.summary.histogram("buckets", bucket_num)
    else:
      # Filter out examples with target_len < 1
      slice_end = tf.cond(features_and_labels["target_len"] >= 1,
                          lambda: tf.constant(1), lambda: tf.constant(0))
      features_and_labels = {
          k: tf.expand_dims(v, 0)[0:slice_end]
          for k, v in features_and_labels.items()
      }
      batch = tf.train.batch(
          tensors=features_and_labels,
          enqueue_many=True,
          batch_size=batch_size,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          name="batch_queue")

    # Separate features and labels again
    features_batch = {k: batch[k] for k in feature_keys}
    labels_batch = {k: batch[k] for k in label_keys}

    return features_batch, labels_batch

  return input_fn
