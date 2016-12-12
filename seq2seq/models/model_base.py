"""Base class for models"""

import tensorflow as tf

from seq2seq import decoders
from seq2seq import losses as seq2seq_losses
from seq2seq.training import featurizers


class ModelBase(object):
  """Abstract base class for models.

  Args:
    params: A dictionary of hyperparameter values
    name: A name for this model to be used as a variable scope
  """

  def __init__(self, params, name):
    self.name = name
    self.params = params

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
        "optimizer.name": "Adam",
        "optimizer.learning_rate": 1e-4,
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

  def _create_predictions(self,
                          features,
                          labels,
                          decoder_output,
                          log_perplexities=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
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
      # Use the embedded prediction as the input to the next time step
      decoder_input_fn_infer = decoders.DynamicDecoderInputs(
          initial_inputs=initial_input,
          make_input_fn=lambda x: tf.nn.embedding_lookup(target_embedding, x.predictions)
      )
      # Decode
      decoder_output, _ = self.encode_decode(
          source=source_embedded,
          source_len=features["source_len"],
          decoder_input_fn=decoder_input_fn_infer,
          target_len=self.params["target.max_seq_len"],
          mode=mode)
      predictions = self._create_predictions(
          features=features, labels=-labels, decoder_output=decoder_output)
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

    # TODO: For a long sequence  logits are a huge [B * T, vocab_size] matrix
    # which can lead to OOM errors on a GPU. Fixing this is TODO, maybe we
    # can use map_fn or slice the logits to max(sequence_length).
    # Should benchmark this.

    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits[:, :-1, :],
        targets=labels["target_ids"][:, 1:],
        sequence_length=labels["target_len"] - 1)

    # Calulate per-example losses of shape [B]
    log_perplexities = tf.div(tf.reduce_sum(
        losses, reduction_indices=1),
                              tf.to_float(labels["target_len"] - 1))

    loss = tf.reduce_mean(log_perplexities)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=self.params["optimizer.learning_rate"],
        clip_gradients=self.params["optimizer.clip_gradients"],
        optimizer=self.params["optimizer.name"],
        summaries=tf.contrib.layers.optimizers.OPTIMIZER_SUMMARIES)

    if mode == tf.contrib.learn.ModeKeys.EVAL:
      train_op = None

    predictions = self._create_predictions(
        features=features,
        labels=labels,
        decoder_output=decoder_output,
        log_perplexities=log_perplexities)

    # We add "useful" tensors to the graph collection so that we
    # can easly find them in our hooks/monitors.
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

    # Summaries
    tf.summary.scalar("loss", loss)

    return predictions, loss, train_op
