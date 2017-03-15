## Decoder Reference

The following tables list available decoder classes and parameters.

### [`BasicDecoder`](https://github.com/google/seq2seq/blob/master/seq2seq/decoders/basic_decoder.py)
---

A Recurrent Neural Network decoder that produces a sequence of output tokens.

| Name | Default | Description |
| --- | --- | --- |
| `max_decode_length` | `100` | Stop decoding early if a sequence reaches this length threshold. |
| `rnn_cell.cell_class` | `BasicLSTMCell` | The class of the rnn cell. Cell classes can be fully defined (e.g. `tensorflow.contrib.rnn.BasicRNNCell`) or must be in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`. |
| `rnn_cell.cell_params` | `{"num_units": 128}` | A dictionary of parameters to pass to the cell class constructor. |
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the encoder. |

### [`AttentionDecoder`](https://github.com/google/seq2seq/blob/master/seq2seq/decoders/attention_decoder.py)
---

A Recurrent Neural Network decoder that produces a sequence of output tokens using an attention mechanisms over its inputs. Parameters are the same as for `BasicDecoder`.
