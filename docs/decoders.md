## Decoder Reference

The following tables list available decoder classes and their hyperparameters.

### `BasicDecoder`

| Name | Default | Description |
| --- | --- | --- |
| `rnn_cell.cell_spec` | `{ "class": "BasicLSTMCell", "num_units": 128}` | A dictioanry that specifies the cell class and parameters, for example `{ "class": "LSTMCell", "num_units": 128, "use_peepholes": true }`. The dictionary object must contain a `class` property as well as arguments that are required by the cell class constructor. Cell classes are assumed to be defined in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`.|
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the encoder. |

### `AttentionDecoder`

Same as `BasicDecoder`.