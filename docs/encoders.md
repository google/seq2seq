## Encoder Reference

All encoders inherit from the abstract `Encoder` defined in `seq2seq.encoders.encoder` and receive `params`, `mode` arguments at instantiation time. Available hyperparameters vary by encoder class.

### `UnidirectionalRNNEncoder`

| Name | Default | Description |
| --- | --- | --- |
| `rnn_cell.cell_class` | `BasicLSTMCell` | The class of the rnn cell. Cell classes can be fully defined (e.g. `tensorflow.contrib.rnn.BasicRNNCell`) or must be in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`. |
| `rnn_cell.cell_params` | `{"num_units": 128}` | A dictionary of parameters to pass to the cell class constructor. |
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the encoder. |

### `BidirectionalRNNEncoder`

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.

### `StackBidirectionalRNNEncoder`

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.
