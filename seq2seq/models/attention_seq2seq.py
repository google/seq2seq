"""
Sequence to Sequence model with attention
"""

import tensorflow as tf

import seq2seq.encoders
import seq2seq.decoders
from seq2seq.models import Seq2SeqBase

class AttentionSeq2Seq(Seq2SeqBase):

  def __init__(self, source_vocab_info, target_vocab_info, params, name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(source_vocab_info, target_vocab_info, params, name)

  @staticmethod
  def default_params():
    params = Seq2SeqBase.default_params().copy()
    params.update({
      "attention.dim": 128,
      "rnn_cell.type": "LSTMCell",
      "rnn_cell.num_units": 128,
      "rnn_cell.dropout_input_keep_prob": 1.0,
      "rnn_cell.dropout_output_keep_prob": 1.0,
      "rnn_cell.num_layers": 1
    })
    return params

  def _encode_decode(self, source, source_len, decoder_input_fn, target_len, labels=None):
    encoder_cell = seq2seq.training.utils.get_rnn_cell(
      cell_type=self.params["rnn_cell.type"],
      num_units=self.params["rnn_cell.num_units"],
      num_layers=self.params["rnn_cell.num_layers"],
      dropout_input_keep_prob=self.params["rnn_cell.dropout_input_keep_prob"],
      dropout_output_keep_prob=self.params["rnn_cell.dropout_output_keep_prob"])
    encoder_fn = seq2seq.encoders.BidirectionalRNNEncoder(encoder_cell)
    encoder_output = encoder_fn(source, source_len)

    decoder_cell = encoder_cell
    decoder_fn = seq2seq.decoders.AttentionDecoder(
      cell=decoder_cell,
      vocab_size=self.target_vocab_info.total_size,
      attention_inputs=encoder_output.outputs,
      attention_fn=seq2seq.decoders.AttentionLayer(self.params["attention.dim"]),
      max_decode_length=target_len)

    decoder_output, _, _ = decoder_fn(
      input_fn=decoder_input_fn,
      initial_state=decoder_cell.zero_state(tf.shape(source_len)[0], dtype=tf.float32),
      sequence_length=target_len)

    if labels is None:
      return decoder_output, None

    assert target_len is not None, "Must pass both labels and target_length"

    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq.losses.cross_entropy_sequence_loss(
      logits=decoder_output.logits[:, :-1],
      targets=labels,
      sequence_length=target_len - 1)

    # Calulate per-example losses of shape [B]
    log_perplexities = tf.div(tf.reduce_sum(
      losses, reduction_indices=1), tf.to_float(target_len))

    return decoder_output, log_perplexities
