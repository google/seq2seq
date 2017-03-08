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
Task where both the input and output sequence are plain text.
"""

import functools
import os

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.data import vocab
from seq2seq.tasks.training_task import TrainingTask
from seq2seq.tasks.inference_task import InferenceTask
from seq2seq.training import hooks

def _get_prediction_length(predictions_dict):
  """Returns the length of the prediction based on the index
  of the first SEQUENCE_END token.
  """
  tokens_iter = enumerate(predictions_dict["predicted_tokens"])
  return next(
      ((i + 1) for i, _ in tokens_iter if _ == "SEQUENCE_END"),
      len(predictions_dict["predicted_tokens"]))

def _get_scores(predictions_dict):
  """Returns the attention scores, sliced by source and target length.
  """
  prediction_len = _get_prediction_length(predictions_dict)
  source_len = predictions_dict["features.source_len"]
  return predictions_dict["attention_scores"][:prediction_len, :source_len]

def _create_figure(predictions_dict):
  """Creates and returns a new figure that visualizes
  attention scores for for a single model predictions.
  """

  # Find out how long the predicted sequence is
  target_words = list(predictions_dict["predicted_tokens"])

  prediction_len = _get_prediction_length(predictions_dict)

  # Get source words
  source_len = predictions_dict["features.source_len"]
  source_words = predictions_dict["features.source_tokens"][:source_len]

  # Plot
  fig = plt.figure(figsize=(8, 8))
  plt.imshow(
      X=predictions_dict["attention_scores"][:prediction_len, :source_len],
      interpolation="nearest",
      cmap=plt.cm.Blues)
  plt.xticks(np.arange(source_len), source_words, rotation=45)
  plt.yticks(np.arange(prediction_len), target_words, rotation=-45)
  fig.tight_layout()

  return fig

def _get_unk_mapping(filename):
  """Reads a file that specifies a mapping from source to target tokens.
  The file must contain lines of the form <source>\t<target>"

  Args:
    filename: path to the mapping file

  Returns:
    A dictionary that maps from source -> target tokens.
  """
  with gfile.GFile(filename, "r") as mapping_file:
    lines = mapping_file.readlines()
    mapping = dict([_.split("\t")[0:2] for _ in lines])
    mapping = {k.strip(): v.strip() for k, v in mapping.items()}
  return mapping

def _unk_replace(source_tokens, predicted_tokens, attention_scores,
                 mapping=None):
  """Replaces UNK tokens with tokens from the source or a
  provided mapping based on the attention scores.

  Args:
    source_tokens: A numpy array of strings.
    predicted_tokens: A numpy array of strings.
    attention_scores: A numeric numpy array
      of shape `[prediction_length, source_length]` that contains
      the attention scores.
    mapping: If not provided, an UNK token is replaced with the
      source token that has the highest attention score. If provided
      the token is insead replaced with `mapping[chosen_source_token]`.

  Returns:
    A new `predicted_tokens` array.
  """
  result = []
  for token, scores in zip(predicted_tokens, attention_scores):
    if token == "UNK":
      max_score_index = np.argmax(scores)
      chosen_source_token = source_tokens[max_score_index]
      new_target = chosen_source_token
      if mapping is not None and chosen_source_token in mapping:
        new_target = mapping[chosen_source_token]
      result.append(new_target)
    else:
      result.append(token)
  return np.array(result)

class TextToTextTrain(TrainingTask):
  """Defines training for tasks where both the input and output sequences
    are plain text.

  Params:
    delimiter_source: Character by which source tokens are delimited.
      Defaults to space.
    delimiter_target: Character by which target tokens are delimited.
      Defaults to space.
    metrics: A list of metrics to be tracked during evaluation.
    train_sample_frequency: Sample generated responses during training every
      N steps.
    vocab_source: Path to vocabulary file used for the source sequence.
    vocab_target: Path to vocabulary file used for the target sequence.
  """
  def __init__(self, params):
    super(TextToTextTrain, self).__init__(params)
    # Load vocabulary info
    self._source_vocab_info = vocab.get_vocab_info(self.params["vocab_source"])
    self._target_vocab_info = vocab.get_vocab_info(self.params["vocab_target"])

  @staticmethod
  def default_params():
    params = TrainingTask.default_params()
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
    return self._model_cls(
        source_vocab_info=self._source_vocab_info,
        target_vocab_info=self._target_vocab_info,
        params=self.params["model_params"],
        mode=mode)

  def create_training_hooks(self, estimator):
    training_hooks = super(
        TextToTextTrain, self).create_training_hooks(estimator)

    output_dir = estimator.model_dir

    train_sample_hook = hooks.TrainSampleHook(
        every_n_steps=self.params["train_sample_frequency"],
        sample_dir=os.path.join(output_dir, "samples"),
        source_delimiter=self.params["delimiter_source"],
        target_delimiter=self.params["delimiter_target"])
    training_hooks.append(train_sample_hook)

    tokens_per_sec_counter = hooks.TokensPerSecondCounter(
        every_n_steps=100,
        output_dir=output_dir)
    training_hooks.append(tokens_per_sec_counter)

    return training_hooks


class TextToTextInfer(InferenceTask):
  """Defines inference for tasks where both the input and output sequences
  are plain text.

  Params:
    delimiter: Character by which tokens are delimited. Defaults to space.
    unk_replace: If true, enable unknown token replacement based on attention
      scores.
    unk_mapping: If `unk_replace` is true, this can be the path to a file
      defining a dictionary to improve UNK token replacement. Refer to the
      documentation for more details.
    dump_attention_dir: Save attention scores and plots to this directory.
    dump_attention_no_plot: If true, only save attention scores, not
      attention plots.
    dump_beams: Write beam search debugging information to this file.
  """

  def __init__(self, params, train_options):
    super(TextToTextInfer, self).__init__(params, train_options)
    self._unk_mapping = None
    self._unk_replace_fn = None
    # Accumulate attention scores in this array.
    # Shape: [num_examples, target_length, input_length]
    self._attention_scores_accum = []
    # Accumulate beam search debug information in these arrays
    self._beam_accum = {
        "predicted_ids": [],
        "beam_parent_ids": [],
        "scores": [],
        "log_probs": []
    }
    # Use vocab from training
    self._source_vocab_info = vocab.get_vocab_info(
        self._train_options.task_params["vocab_source"])
    self._target_vocab_info = vocab.get_vocab_info(
        self._train_options.task_params["vocab_target"])

  @staticmethod
  def default_params():
    params = InferenceTask.default_params()
    params.update({
        "delimiter": " ",
        "unk_replace": False,
        "unk_mapping": None,
        "dump_attention_dir": None,
        "dump_attention_no_plot": None,
        "dump_beams": None,
    })
    return params

  def create_model(self):
    return self._model_cls(
        source_vocab_info=self._source_vocab_info,
        target_vocab_info=self._target_vocab_info,
        params=self.params["model_params"],
        mode=tf.contrib.learn.ModeKeys.INFER)

  def prediction_keys(self):
    prediction_keys = set([
        "predicted_tokens", "features.source_len", "features.source_tokens",
        "attention_scores"])
    if self.params["dump_beams"] is not None:
      prediction_keys.update([
          "beam_search_output.predicted_ids",
          "beam_search_output.beam_parent_ids",
          "beam_search_output.scores",
          "beam_search_output.log_probs"])
    if self.params["unk_replace"]:
      prediction_keys.add("attention_scores")
    return prediction_keys

  def begin(self):
    if self.params["unk_mapping"] is not None:
      self._unk_mapping = _get_unk_mapping(self.params["unk_mapping"])
    if self.params["unk_replace"]:
      self._unk_replace_fn = functools.partial(
          _unk_replace, mapping=self._unk_mapping)

    if self.params["dump_attention_dir"] is not None:
      gfile.MakeDirs(self.params["dump_attention_dir"])

  def process_batch(self, idx, predictions_dict):
    # Convert to unicode
    predictions_dict["predicted_tokens"] = np.char.decode(
        predictions_dict["predicted_tokens"].astype("S"), "utf-8")
    predicted_tokens = predictions_dict["predicted_tokens"]

    # If we're using beam search we take the first beam
    if np.ndim(predicted_tokens) > 1:
      predicted_tokens = predicted_tokens[:, 0]

    predictions_dict["features.source_tokens"] = np.char.decode(
        predictions_dict["features.source_tokens"].astype("S"), "utf-8")
    source_tokens = predictions_dict["features.source_tokens"]
    source_len = predictions_dict["features.source_len"]

    if self._unk_replace_fn is not None:
      # We slice the attention scores so that we do not
      # accidentially replace UNK with a SEQUENCE_END token
      attention_scores = predictions_dict["attention_scores"]
      attention_scores = attention_scores[:, :source_len - 1]
      predicted_tokens = self._unk_replace_fn(
          source_tokens=source_tokens,
          predicted_tokens=predicted_tokens,
          attention_scores=attention_scores)

    # Optionally Dump attention
    if self.params["dump_attention_dir"] is not None:
      if not self.params["dump_attention_no_plot"]:
        output_path = os.path.join(
            self.params["dump_attention_dir"], "{:05d}.png".format(idx))
        _create_figure(predictions_dict)
        plt.savefig(output_path)
        plt.close()
        tf.logging.info("Wrote %s", output_path)
      self._attention_scores_accum.append(_get_scores(predictions_dict))

    # Optionally dump beams
    if self.params["dump_beams"] is not None:
      self._beam_accum["predicted_ids"] += [predictions_dict[
          "beam_search_output.predicted_ids"]]
      self._beam_accum["beam_parent_ids"] += [predictions_dict[
          "beam_search_output.beam_parent_ids"]]
      self._beam_accum["scores"] += [predictions_dict[
          "beam_search_output.scores"]]
      self._beam_accum["log_probs"] += [predictions_dict[
          "beam_search_output.log_probs"]]

    sent = self.params["delimiter"].join(
        predicted_tokens).split("SEQUENCE_END")[0]
    # Replace special BPE tokens
    sent = sent.replace("@@ ", "")
    sent = sent.strip()

    print(sent)

  def end(self):
    # Write attention scores
    if self.params["dump_attention_dir"] is not None:
      scores_path = os.path.join(
          self.params["dump_attention_dir"], "attention_scores.npz")
      np.savez(scores_path, *self._attention_scores_accum)

    # Write beams
    if self.params["dump_beams"] is not None:
      np.savez(self.params["dump_beams"], **self._beam_accum)
