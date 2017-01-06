"""Miscellaneous training utility functions.
"""

import os

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.data.data_utils import read_from_data_provider
from seq2seq.training import hooks

def get_rnn_cell(cell_type,
                 num_units,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0):
  """Creates a new RNN Cell.

  Args:
    cell_type: A cell lass name defined in `tf.contrib.rnn`,
      e.g. `LSTMCell` or `GRUCell`
    num_units: Number of cell units
    num_layers: Number of layers. The cell will be wrapped with
      `tf.contrib.rnn.MultiRNNCell`
    dropout_input_keep_prob: Dropout keep probability applied
      to the input of cell *at each layer*
    dropout_output_keep_prob: Dropout keep probability applied
      to the output of cell *at each layer*

  Returns:
    An instance of `tf.contrib.rnn.RNNCell`.
  """
  #pylint: disable=redefined-variable-type
  cell_class = getattr(tf.contrib.rnn, cell_type)
  cell = cell_class(num_units)

  if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
    cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell,
        input_keep_prob=dropout_input_keep_prob,
        output_keep_prob=dropout_output_keep_prob)

  if num_layers > 1:
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

  return cell


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False):
  """Creates a function that decays the learning rate.

  Args:
    decay_steps: How often to apply decay.
    decay_rate: A Python number. The decay rate.
    start_decay_at: Don't decay before this step
    stop_decay_at: Don't decay after this step
    min_learning_rate: Don't decay below this number
    decay_type: A decay function name defined in `tf.train`
    staircase: Whether to apply decay in a discrete staircase,
      as opposed to continuous, fashion.

  Returns:
    A function that takes (learning_rate, global_step) as inputs
    and returns the learning rate for the given step.
    Returns `None` if decay_type is empty or None.
  """
  if decay_type is None or decay_type == "":
    return None

  def decay_fn(learning_rate, global_step):
    """The computed learning rate decay function.
    """
    decay_type_fn = getattr(tf.train, decay_type)
    decayed_learning_rate = decay_type_fn(
        learning_rate=learning_rate,
        global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
        name="decayed_learning_rate")

    final_lr = tf.train.piecewise_constant(
        x=global_step,
        boundaries=[start_decay_at],
        values=[learning_rate, decayed_learning_rate])

    if min_learning_rate:
      final_lr = tf.maximum(final_lr, min_learning_rate)

    return final_lr

  return decay_fn


def create_input_fn(data_provider_fn,
                    batch_size,
                    bucket_boundaries=None,
                    allow_smaller_final_batch=False):
  """Creates an input function that can be used with tf.learn estimators.
    Note that you must pass "factory funcitons" for both the data provider and
    featurizer to ensure that everything will be created in  the same graph.

  Args:
    data_provider_fn: Function that creates a data provider to read from.
      An instance of `tf.contrib.slim.data_provider.DataProvider`.
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
    features_and_labels = read_from_data_provider(data_provider_fn())

    if bucket_boundaries:
      bucket_num, batch = tf.contrib.training.bucket_by_sequence_length(
          input_length=features_and_labels["source_len"],
          bucket_boundaries=bucket_boundaries,
          tensors=features_and_labels,
          batch_size=batch_size,
          keep_input=features_and_labels["source_len"] >= 1,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          allow_smaller_final_batch=allow_smaller_final_batch,
          name="bucket_queue")
      tf.summary.histogram("buckets", bucket_num)
    else:
      batch = tf.train.batch(
          tensors=features_and_labels,
          enqueue_many=False,
          batch_size=batch_size,
          dynamic_pad=True,
          capacity=5000 + 16 * batch_size,
          allow_smaller_final_batch=allow_smaller_final_batch,
          name="batch_queue")

    # Separate features and labels
    features_batch = {k: batch[k] for k in ("source_tokens", "source_len")}
    if "target_tokens" in batch:
      labels_batch = {k: batch[k] for k in ("target_tokens", "target_len")}
    else:
      labels_batch = None

    return features_batch, labels_batch

  return input_fn


def write_hparams(hparams_dict, path):
  """
  Writes hyperparameter values to a file.

  Args:
    hparams_dict: The dictionary of hyperparameters
    path: Absolute path to write to
  """
  gfile.MakeDirs(os.path.dirname(path))
  out = "\n".join(
      ["{}={}".format(k, v) for k, v in sorted(hparams_dict.items())])
  with gfile.GFile(path, "w") as file:
    file.write(out)


def read_hparams(path):
  """
  Reads hyperparameters into a string that can be used with a
  HParamsParser.

  Args:
    path: Absolute path to the file to read from
  """
  with gfile.GFile(path, "r") as file:
    lines = file.readlines()
  return ",".join(lines)


def create_default_training_hooks(estimator, sample_frequency=500):
  """Creates common SessionRunHooks used for training.

  Args:
    estimator: The estimator instance
    sample_frequency: frequency of samples passed to the TrainSampleHook

  Returns:
    An array of `SessionRunHook` items.
  """
  output_dir = estimator.model_dir
  training_hooks = []

  model_analysis_hook = hooks.PrintModelAnalysisHook(
      filename=os.path.join(output_dir, "model_analysis.txt"))
  training_hooks.append(model_analysis_hook)

  train_sample_hook = hooks.TrainSampleHook(
      every_n_steps=sample_frequency,
      file=os.path.join(output_dir, "samples.txt"))
  training_hooks.append(train_sample_hook)

  metadata_hook = hooks.MetadataCaptureHook(
      output_dir=os.path.join(output_dir, "metadata"),
      step=10)
  training_hooks.append(metadata_hook)

  tokens_per_sec_counter = hooks.TokensPerSecondCounter(
      every_n_steps=100,
      output_dir=output_dir)
  training_hooks.append(tokens_per_sec_counter)

  return training_hooks

def print_hparams(hparams):
  """Prints hyperparameter values in sorted order.

  Args:
    hparams: A dictionary of hyperparameters.
  """
  tf.logging.info("=" * 50)
  for param, value in sorted(hparams.items()):
    tf.logging.info("%s=%s", param, value)
  tf.logging.info("=" * 50)
