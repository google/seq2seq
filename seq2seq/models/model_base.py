"""Base class for models"""

import tensorflow as tf
import seq2seq

class ModelBase(object):
  def __init__(self, params, name):
    self.name = name
    self.params = params

  @staticmethod
  def default_params():
    """Returns a dictionary of default parameters for this model."""
    return {}

  @classmethod
  def from_params(cls, params):
    """Create a new instance of this class based on a parameter dictionary.

    Args:
      params: A dictionary with string keys that contains the parameters for the model.
       These will be added to the default parameters.

    Returns:
      An instance of this class.
    """
    final_params = cls.default_params().copy()
    final_params.update(params)
    return cls._from_params(final_params)

  @staticmethod
  def _from_params(params):
    """Should be implemented by child classes. See `from_params`"""
    raise NotImplementedError

  def __call__(self, features, labels, params, mode):
    with tf.variable_scope(self.name):
      return self._build(features, labels, params, mode)

  def _build(self, features, labels, params, mode):
    raise NotImplementedError


class Seq2SeqBase(ModelBase):
  """Base class for seq2seq models with embeddings
  """
  def __init__(self, source_vocab_info, target_vocab_info, params, name):
    super(Seq2SeqBase, self).__init__(params, name)
    self.source_vocab_info = source_vocab_info
    self.target_vocab_info = target_vocab_info

  @staticmethod
  def default_params():
    return {
      "decoder.max_seq_len": 40,
      "embedding.dim": 100,
      "optimizer.name": "Adam",
      "optimizer.learning_rate": 1e-4,
      "optimizer.clip_gradients": 5.0,
    }

  def _encode_decode(self, source, source_len, decoder_input_fn, target_len, labels=None):
    """Should be implemented by child classes"""
    raise NotImplementedError

  def _create_predictions(self, features, labels, decoder_output, log_perplexities, mode):
    predictions = {
      "logits": decoder_output.logits,
      "predictions": decoder_output.predictions,
    }
    if log_perplexities is not None:
      predictions["log_perplexities"] = log_perplexities
    return predictions


  def _build(self, features, labels, params, mode):
    # Create embedddings
    source_embedding = tf.get_variable(
      "source_embedding", [self.source_vocab_info.total_size, self.params["embedding.dim"]])
    target_embedding = tf.get_variable(
      "target_embedding", [self.target_vocab_info.total_size, self.params["embedding.dim"]])

    # Embed source
    source_embedded = tf.nn.embedding_lookup(source_embedding, features["source_ids"])

    # Graph used for inference
    if mode == tf.contrib.learn.ModeKeys.INFER:
      target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
      # Embed the "SEQUENCE_START" token
      initial_input = tf.nn.embedding_lookup(
        target_embedding, tf.ones_like(features["source_len"]) * target_start_id)
      # Use the embedded prediction as the input to the next time step
      decoder_input_fn_infer = seq2seq.decoders.DynamicDecoderInputs(
        initial_inputs=initial_input,
        make_input_fn=lambda x: tf.nn.embedding_lookup(target_embedding, x.predictions))
      # Decode
      decoder_output, _ = self._encode_decode(
        source=source_embedded,
        source_len=features["source_len"],
        decoder_input_fn=decoder_input_fn_infer,
        target_len=self.params["decoder.max_seq_len"])
      predictions = self._create_predictions(features, labels, decoder_output, None, mode)
      return predictions, None, None

    # Embed target
    target_embedded = tf.nn.embedding_lookup(target_embedding, labels["target_ids"])

    # During training/eval, we have labels and use them for teacher forcing
    decoder_input_fn_train = seq2seq.decoders.FixedDecoderInputs(
      inputs=target_embedded[:, :-1],
      sequence_length=labels["target_len"] - 1)

    decoder_output, log_perplexities = self._encode_decode(
      source=source_embedded,
      source_len=features["source_len"],
      decoder_input_fn=decoder_input_fn_train,
      target_len=labels["target_len"],
      labels=labels["target_ids"][:, 1:])

    loss = tf.reduce_mean(log_perplexities)

    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=self.params["optimizer.learning_rate"],
      clip_gradients=self.params["optimizer.clip_gradients"],
      optimizer=self.params["optimizer.name"])

    if mode == tf.contrib.learn.ModeKeys.EVAL:
      train_op = None

    predictions = self._create_predictions(features, labels, decoder_output, log_perplexities, mode)

    # We use this collection in our monitors to print samples
    # TODO: Is there a cleaner way to do this?
    for key, tensor in predictions.items():
      tf.add_to_collection("model_output_keys", key)
      tf.add_to_collection("model_output_values", tensor)

    for key, tensor in features.items():
      tf.add_to_collection("features_keys", key)
      tf.add_to_collection("features_values", tensor)

    for key, tensor in labels.items():
      tf.add_to_collection("labels_keys", key)
      tf.add_to_collection("labels_values", tensor)

    return predictions, loss, train_op
