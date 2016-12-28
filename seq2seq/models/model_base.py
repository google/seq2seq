"""Base class for models"""

import collections
import tensorflow as tf

from seq2seq import decoders
from seq2seq import graph_utils
from seq2seq import losses as seq2seq_losses
from seq2seq.decoders.beam_search_decoder import BeamSearchDecoder
from seq2seq.inference import beam_search
from seq2seq.models import featurizers
from seq2seq.training import utils as training_utils

def time_to_batch(tensor, name=None):
  """Transposes the first two dimensions of a tensor. Leaves the remaining
  dimensions unchanged.

  Args:
    tensor: Input tensor to be transposed.

  Returns:
    A tensor of the same type as `tensor` with the first two dimensions
    swapped.
  """
  ndims = tensor.get_shape().ndims
  perm = [1, 0] + list(range(ndims))[2:]
  return tf.transpose(tensor, perm, name=name)


def _flatten_dict(dict_, parent_key="", sep="."):
  """Flattens a nested dictionary. Namedtuples within
  the dictionary are converted to dicts.

  Args:
    dict_: The dictionary to flatten.
    parent_key: A prefix to prepend to each key.
    sep: Separator between parent and child keys, a string. For example
      { "a": { "b": 3 } } will become { "a.b": 3 } if the separator is ".".

  Returns:
    A new flattened dictionary.
  """
  items = []
  for key, value in dict_.items():
    new_key = parent_key + sep + key if parent_key else key
    if isinstance(value, collections.MutableMapping):
      items.extend(_flatten_dict(value, new_key, sep=sep).items())
    elif isinstance(value, tuple) and hasattr(value, "_asdict"):
      items.extend(_flatten_dict(value._asdict(), new_key, sep=sep).items())
    else:
      items.append((new_key, value))
  return dict(items)


class ModelBase(object):
  """Abstract base class for models.

  Args:
    params: A dictionary of hyperparameter values
    name: A name for this model to be used as a variable scope
  """

  def __init__(self, params, name):
    self.name = name
    self.params = params

    # Cast parameters to correct types
    default_params = self.default_params()
    for key, value in self.params.items():
      self.params[key] = type(default_params[key])(value)

  def create_featurizer(self):
    """"Returns a new featurizer instance to be used by this model"""
    raise NotImplementedError

  @staticmethod
  def default_params():
    """Returns a dictionary of default parameters for this model."""
    return {}

  def __call__(self, features, labels, params, mode):
    """Creates the model graph. See the model_fn documentation in
    tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    with tf.variable_scope("model"):
      with tf.variable_scope(self.name):
        return self._build(features, labels, params, mode)

  def _build(self, features, labels, params, mode):
    """Subclasses should implement this method. See the `model_fn` documentation
    in tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    raise NotImplementedError


class Seq2SeqBase(ModelBase):
  """Base class for seq2seq models with embeddings

  TODO: Do we really need to pass source/target vocab info here? It seems ugly.
  It's mostly used to define the output size of the decoder.
  Maybe we can somehow put it in the features?
  """

  def __init__(self, source_vocab_info, target_vocab_info, params, name):
    super(Seq2SeqBase, self).__init__(params, name)
    self.source_vocab_info = source_vocab_info
    self.target_vocab_info = target_vocab_info

  def create_featurizer(self):
    return featurizers.Seq2SeqFeaturizer(
        source_vocab_info=self.source_vocab_info,
        target_vocab_info=self.target_vocab_info,
        max_seq_len_source=self.params["source.max_seq_len"],
        max_seq_len_target=self.params["target.max_seq_len"])

  @staticmethod
  def default_params():
    return {
        "source.max_seq_len": 40,
        "target.max_seq_len": 40,
        "embedding.dim": 100,
        "inference.beam_search.beam_width": 0,
        "inference.beam_search.score_fn": "logprob_score",
        "inference.beam_search.choose_successors_fn": "choose_top_k",
        "optimizer.name": "Adam",
        "optimizer.learning_rate": 1e-4,
        "optimizer.lr_decay_type": "",
        "optimizer.lr_decay_steps": 100,
        "optimizer.lr_decay_rate": 0.99,
        "optimizer.lr_start_decay_at": 0,
        "optimizer.lr_stop_decay_at": 1e9,
        "optimizer.lr_min_learning_rate": 1e-12,
        "optimizer.lr_staircase": False,
        "optimizer.clip_gradients": 5.0,
    }

  def encode_decode(self,
                    source,
                    source_len,
                    decoder_input_fn,
                    target_len,
                    mode=tf.contrib.learn.ModeKeys.TRAIN):
    """Should be implemented by child classes"""
    raise NotImplementedError

  def _create_predictions(self, decoder_output, losses=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
    predictions = _flatten_dict(decoder_output._asdict())

    if losses is not None:
      predictions["losses"] = losses

    # Decoders returns output in time-major form [T, B, ...]
    # Here we transpose everything back to batch-major for the user
    predictions = {k: time_to_batch(v) for k, v in  predictions.items()}

    return predictions

  def _get_beam_search_decoder(self, decoder):
    """Wraps a decoder into a Beam Search decoder.

    Args:
      decoder: The original decoder

    Returns:
      A BeamSearchDecoder with the same interfaces as the original decoder.
    """
    config = beam_search.BeamSearchConfig(
        beam_width=self.params["inference.beam_search.beam_width"],
        vocab_size=self.target_vocab_info.total_size,
        eos_token=self.target_vocab_info.special_vocab.SEQUENCE_END,
        score_fn=getattr(beam_search,
                         self.params["inference.beam_search.score_fn"]),
        choose_successors_fn=getattr(
            beam_search,
            self.params["inference.beam_search.choose_successors_fn"]))
    return BeamSearchDecoder(decoder=decoder, config=config)

  @property
  def use_beam_search(self):
    """Returns true iff the model should perform beam search.
    """
    return self.params["inference.beam_search.beam_width"] > 1

  def _build(self, features, labels, params, mode):
    # Pre-process features and labels
    features, labels = self.create_featurizer()(features, labels)

    # Create embedddings
    source_embedding = tf.get_variable(
        "source_embedding",
        [self.source_vocab_info.total_size, self.params["embedding.dim"]])
    target_embedding = tf.get_variable(
        "target_embedding",
        [self.target_vocab_info.total_size, self.params["embedding.dim"]])

    # Embed source
    source_embedded = tf.nn.embedding_lookup(source_embedding,
                                             features["source_ids"])

    # Graph used for inference
    if mode == tf.contrib.learn.ModeKeys.INFER:
      target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
      # Embed the "SEQUENCE_START" token
      initial_input = tf.nn.embedding_lookup(
          target_embedding,
          tf.ones_like(features["source_len"]) * target_start_id)

      def make_input_fn(predictions):
        """Use the embedded prediction as the input to the next time step
        """
        return tf.nn.embedding_lookup(target_embedding, predictions)

      decoder_input_fn_infer = decoders.DynamicDecoderInputs(
          initial_inputs=initial_input, make_input_fn=make_input_fn)

      # Decode
      decoder_output = self.encode_decode(
          source=source_embedded,
          source_len=features["source_len"],
          decoder_input_fn=decoder_input_fn_infer,
          target_len=self.params["target.max_seq_len"],
          mode=mode)
      predictions = self._create_predictions(
          decoder_output=decoder_output)
      return predictions, None, None

    # Embed target
    target_embedded = tf.nn.embedding_lookup(target_embedding,
                                             labels["target_ids"])

    # During training/eval, we have labels and use them for teacher forcing
    # We don't feed the last SEQUENCE_END token
    decoder_input_fn_train = decoders.FixedDecoderInputs(
        inputs=target_embedded[:, :-1],
        sequence_length=labels["target_len"] - 1)

    decoder_output = self.encode_decode(
        source=source_embedded,
        source_len=features["source_len"],
        decoder_input_fn=decoder_input_fn_train,
        target_len=labels["target_len"],
        mode=mode)

    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits[:-1, :, :],
        targets=tf.transpose(labels["target_ids"][:, 1:], [1, 0]),
        sequence_length=labels["target_len"] - 1)

    # Calculate the average log perplexity
    loss = tf.reduce_sum(losses) / tf.to_float(
        tf.reduce_sum(labels["target_len"] - 1))

    learning_rate_decay_fn = training_utils.create_learning_rate_decay_fn(
        decay_type=self.params["optimizer.lr_decay_type"] or None,
        decay_steps=self.params["optimizer.lr_decay_steps"],
        decay_rate=self.params["optimizer.lr_decay_rate"],
        start_decay_at=self.params["optimizer.lr_start_decay_at"],
        stop_decay_at=self.params["optimizer.lr_stop_decay_at"],
        min_learning_rate=self.params["optimizer.lr_min_learning_rate"],
        staircase=self.params["optimizer.lr_staircase"])

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=self.params["optimizer.learning_rate"],
        learning_rate_decay_fn=learning_rate_decay_fn,
        clip_gradients=self.params["optimizer.clip_gradients"],
        optimizer=self.params["optimizer.name"],
        summaries=tf.contrib.layers.optimizers.OPTIMIZER_SUMMARIES)

    if mode == tf.contrib.learn.ModeKeys.EVAL:
      train_op = None

    predictions = self._create_predictions(
        decoder_output=decoder_output,
        losses=losses)

    # We add "useful" tensors to the graph collection so that we
    # can easly find them in our hooks/monitors.
    graph_utils.add_dict_to_collection(predictions, "model_output")
    graph_utils.add_dict_to_collection(features, "features")
    graph_utils.add_dict_to_collection(labels, "labels")

    # Summaries
    tf.summary.scalar("loss", loss)

    return predictions, loss, train_op
