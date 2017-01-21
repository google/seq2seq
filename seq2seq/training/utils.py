# -*- coding: utf-8 -*-

"""Miscellaneous training utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import inspect
import os
import re
import subprocess
import tempfile
import json

import numpy as np
from six.moves import urllib

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.contrib import rnn_cell
from seq2seq.data.data_utils import read_from_data_provider
from seq2seq.training import hooks

class TrainOptions(object):
  """A collectionf of options that are passed to the training script
  and should be saved to perform inference later on.

  Args:
    model_dir: The model directory. Options will be dumped in this
      directory.
    hparams: A dictionary of hyperparameter values.
    model_class: The model class name, a string.
    source_vocab_path: Path to the source vocabulary
    target_vocab_path: Path to the target vocabulary
  """
  def __init__(self,
               hparams=None,
               model_class=None,
               source_vocab_path=None,
               target_vocab_path=None):
    self.hparams = hparams
    self.model_class = model_class
    self.source_vocab_path = source_vocab_path
    self.target_vocab_path = target_vocab_path

  @staticmethod
  def path(model_dir):
    """Returns the path to the options file.

    Args:
      model_dir: The model directory
    """
    return os.path.join(model_dir, "train_options.json")

  def dump(self, model_dir):
    """Dumps the options to a file in the model directory.

    Args:
      model_dir: Path to the model directory. The options will be
      dumped into a file in this directory.
    """
    gfile.MakeDirs(model_dir)
    options_dict = {
        "hparams": self.hparams,
        "model_class": self.model_class,
        "source_vocab_path": self.source_vocab_path,
        "target_vocab_path": self.target_vocab_path
    }

    with gfile.GFile(TrainOptions.path(model_dir), "w") as file:
      file.write(json.dumps(options_dict).encode("utf-8"))

  @staticmethod
  def load(model_dir):
    """ Loads options from the given model directory.

    Args:
      model_dir: Path to the model directory.
    """
    with gfile.GFile(TrainOptions.path(model_dir), "r") as file:
      options_dict = json.loads(file.read().decode("utf-8"))
    options_dict = defaultdict(None, options_dict)

    return TrainOptions(
        hparams=options_dict["hparams"],
        model_class=options_dict["model_class"],
        source_vocab_path=options_dict["source_vocab_path"],
        target_vocab_path=options_dict["target_vocab_path"])

def cell_from_spec(cell_spec):
  """Create a RNN Cell instance from a JSON string.

  Args:
    cell_spec: A JSON string of the form
      { "class": "BasicLSTMCell", "num_units": 16, ... }
      The "class" property is treated in a special way and used
      to look up the class object in `seq2seq.contrib.rnn_cell`. All other
      items in the JSON object are passed as parameters to the cell
      constructor.

  Returns:
    A RNNCell instance.
  """

  cell_spec = cell_spec.copy()

  if "class" not in cell_spec:
    raise ValueError("cell_spec must specify \"class\".")

  # Find the cell class
  cell_class_name = cell_spec.pop("class")
  cell_class = getattr(rnn_cell, cell_class_name)

  # Make sure additional arguments are valid
  cell_args = set(inspect.getargspec(cell_class.__init__).args[1:])
  for key in cell_spec.keys():
    if key not in cell_args:
      raise ValueError(
          """{} is not a valid argument for {} class. Available arguments
          are: {}""".format(key, cell_class.__name__, cell_args))

  # Create cell
  return cell_class(**cell_spec)


def get_rnn_cell(cell_spec,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
  """Creates a new RNN Cell.

  Args:
    cell_spec: A JSON string that defines how to create a cell instance.
      See `cell_from_spec` for more details.
    num_layers: Number of layers. The cell will be wrapped with
      `tf.contrib.rnn.MultiRNNCell`
    dropout_input_keep_prob: Dropout keep probability applied
      to the input of cell *at each layer*
    dropout_output_keep_prob: Dropout keep probability applied
      to the output of cell *at each layer*
    residual_connections: If true, add residual connections
      between all cells

  Returns:
    An instance of `tf.contrib.rnn.RNNCell`.
  """
  #pylint: disable=redefined-variable-type
  cell = cell_from_spec(cell_spec)

  if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
    cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell,
        input_keep_prob=dropout_input_keep_prob,
        output_keep_prob=dropout_output_keep_prob)

  if num_layers > 1:
    cell = rnn_cell.ExtendedMultiRNNCell(
        cells=[cell] * num_layers,
        residual_connections=residual_connections,
        residual_combiner=residual_combiner,
        residual_dense=residual_dense)

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


def create_default_training_hooks(estimator, sample_frequency=500,
                                  delimiter=" "):
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
      sample_dir=os.path.join(output_dir, "samples"),
      delimiter=delimiter)
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


def moses_multi_bleu(hypotheses,
                     references,
                     lowercase=False,
                     eos_token="SEQUENCE_END"):
  """Calculate the bleu score for hypotheses and references
  using the MOSES ulti-bleu.perl script.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    eos_token: Slice hypotheses and references up to this token

  Returns:
    The BLEU score as a float32 value.
  """

  if np.size(hypotheses) == 0:
    return np.float32(0.0)

  # Get MOSES multi-bleu script
  multi_bleu_path, _ = urllib.request.urlretrieve(
      "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
      "master/scripts/generic/multi-bleu.perl")
  os.chmod(multi_bleu_path, 0o755)

  # Alternatively, get file locally
  # training_dir = os.path.dirname(os.path.realpath(__file__))
  # bin_dir = os.path.abspath(os.path.join(training_dir, "..", "..", "bin"))
  # multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

  # Deal with byte chars
  if hypotheses.dtype.kind == np.dtype("U"):
    hypotheses = np.char.encode(hypotheses, "utf-8")
  if references.dtype.kind == np.dtype("U"):
    references = np.char.encode(references, "utf-8")

  # Slice all hypotheses and references up to EOS
  sliced_hypotheses = [x.split(eos_token.encode("utf-8"))[0].strip()
                       for x in hypotheses]
  sliced_references = [x.split(eos_token.encode("utf-8"))[0].strip()
                       for x in references]

  # Strip special "@@ " tokens used for BPE
  # See https://github.com/rsennrich/subword-nmt
  # We hope this is rare enough that it will not have any adverse effects
  # on predicitons that do not use BPE
  sliced_hypotheses = [_.replace(b"@@ ", b"") for _ in sliced_hypotheses]
  sliced_references = [_.replace(b"@@ ", b"") for _ in sliced_references]

  # Dump hypotheses and references to tempfiles
  hypothesis_file = tempfile.NamedTemporaryFile()
  hypothesis_file.write(b"\n".join(sliced_hypotheses))
  hypothesis_file.write(b"\n")
  hypothesis_file.flush()
  reference_file = tempfile.NamedTemporaryFile()
  reference_file.write(b"\n".join(sliced_references))
  reference_file.write(b"\n")
  reference_file.flush()

  # Calculate BLEU using multi-bleu script
  with open(hypothesis_file.name, "r") as read_pred:
    bleu_cmd = [multi_bleu_path]
    if lowercase:
      bleu_cmd += ["-lc"]
    bleu_cmd += [reference_file.name]
    bleu_out = subprocess.check_output(
        bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
    bleu_out = bleu_out.decode("utf-8")
    bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
    bleu_score = float(bleu_score)

  # Close temp files
  hypothesis_file.close()
  reference_file.close()

  return np.float32(bleu_score)
