## Encoder Reference

All encoders inherit from the abstract `Encoder` defined in `seq2seq.encoders.encoder` and receive `params`, `mode` arguments at instantiation time. Available hyperparamters vary from encoder to encoder class.

### `UnidirectionalRNNEncoder`

| Name | Default | Description |
| --- | --- | --- |
| `rnn_cell.cell_spec` | `{ "class": "BasicLSTMCell", "num_units": 128}` | A dictioanry that specifies the cell class and parameters, for example `{ "class": "LSTMCell", "num_units": 128, "use_peepholes": true }`. The dictionary object must contain a `class` property as well as arguments that are required by the cell class constructor. Cell classes are assumed to be defined in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`.|
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the encoder. |

### `BidirectionalRNNEncoder`

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.

### `StackBidirectionalRNNEncoder`

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.
