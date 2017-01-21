When calling the training script, you can specify a model class using the `--model` flags and model-specific hyperparameters using the `--hparams` flag. This page lists all supported models and hyperparameters.

## Common Hyperparameters

The following hyperparameters are shared by all models, unless explicitly stated otherwise in the model section.

| Name | Default | Description |
| --- | --- | --- |
| `source.max_seq_len` | `40` | Maximum length of source sequences. An example is sliced to this length before being fed to the encoder. |
| `source.reverse` | `True` | If set to true, reverse the source sequence before feeding it into the encoder.|
| `target.max_seq_len` | `40` | Maximum length of target sequences. An example is sliced to this length before being fed to the decoder. |
| `embedding.dim` | `100` | Dimensionality of the embedding layer. |
| `inference.max_decode_length` | `100` | During inference mode, decode up to this length or until a `SEQUENCE_END` token is encountered, whichever happens first. |
| `optimizer.name` | `Adam` | Type of Optimizer to use, e.g. `Adam`, `SGD` or `Momentum`. The name is fed to TensorFlow's [optimize_loss](https://www.tensorflow.org/api_docs/python/contrib.layers/optimization#optimize_loss) function. See TensorFlow documentation for more details and all available options. |
| `optimizer.learning_rate` | `1e-4` | Initial learning rate for the optimizer. This is fed to TensorFlow's [optimize_loss](https://www.tensorflow.org/api_docs/python/contrib.layers/optimization#optimize_loss) function. |
| `optimizer.lr_decay_type` |  | The name of one of TensorFlow's [learning rate decay functions](https://www.tensorflow.org/api_docs/python/#training--decaying-the-learning-rate) defined in `tf.train`, e.g. `exponential_decay`. If this is an empty string (default) then no learning rate decay is used. |
| `optimizer.lr_decay_steps` | `100` | How often to apply decay. This is fed as the `decay_steps` argument to the decay function defined above. See Tensoflow documentation for more details. |
| `optimizer.lr_decay_rate` | `0.99` | The decay rate. This is fed as the `decay_rate` argument to the decay function defined above. See TensorFlow documentation for more details. |
| `optimizer.lr_start_decay_at` | `0` | Start learning rate decay at this step. |
| `optimizer.lr_stop_decay_at` | `1e9` | Stop learning rate decay at this step.  |
| `optimizer.lr_min_learning_rate` | `1e-12` | Never decay below this learning rate. |
| `optimizer.lr_staircase` | `False` | If `True`, decay the learning rate at discrete intervals. This is fed as the `staircase` argument to the decay function defined above. See TensorFlow documentation for more details. |
| `optimizer.clip_gradients` | `5.0` | Clip gradients by their global norm. |

## BasicSeq2Seq

The `BasicSeq2Seq` model uses an encoder and decoder with no attention mechanism. The last encoder state is passed through a fully connected layer and used to initialize the decoder (this behavior can be changed using the `bridge_spec` hyperparameter). This is the "vanilla" implementation of the seq2seq architecture. The model supports the following additional hyperparameters.

| Name | Default | Description |
| --- | --- | --- |
| `bridge_spec` | `{ "class": "InitialStateBridge"}` | A dictionary that defines how state is passed between encoder and decoder. The `class` property corresponds to a name of bridge class defined in `seq2seq.models.bridges`. All additional properties in the dictinary are passed to the bridge class constructor, e.g. `{"class": "InitialStateBridge", "activation_fn": "tanh"}`. |
| `encoder.type` | `UnidirectionalRNNEncoder` | Type of encoder to use. This is the class name of an encoder defined in `seq2seq.encoder`. Currently the supported value are `BidirectionalRNNEncoder`, `UnidirectionalRNNEncoder` and `StackBidirectionalRNNEncoder`. |
| `encoder.rnn_cell.cell_spec` | `{ "class": "BasicLSTMCell", "num_units": 128}` | A dictioanry that specifies the cell class and parameters, for example `{ "class": "LSTMCell", "num_units": 128, "use_peepholes": true }`. The dictionary object must contain a `class` property as well as arguments that are required by the cell class constructor. Cell classes are assumed to be defined in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`.|
| `encoder.rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `encoder.rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `encoder.rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `encoder.rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the encoder. |
| `decoder.rnn_cell.cell_spec` | `{ "class": "BasicLSTMCell", "num_units": 128}` | Same as `encoder.rnn_cell.cell_spec`, but for the decoder cell. |
| `decoder.rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `decoder.rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `decoder.rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `decoder.rnn_cell.residual_connections` | `False` | If true, add residual connections between all RNN layers in the decoder. |



## AttentionSeq2seq

`AttentionSeq2seq` is a sequence to sequence model that uses a unidirectional or bidirectional encoder and a decoder with an attention mechanism. By default, the last encoder state is not fed to the decoder. This implementation is comparable to the model in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This model supports the same parameters as the `BasicSeq2Seq` model, plus the following:

| Name | Default | Description |
| --- | --- | --- |
| `attention.dim` | `128` | Number of units in the attention layer. |
| `attention.score_type` | `dot` | The formula used to calculate attention scores. Available values are `bahdanau` and `dot`. `bahdanau` is described in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). `dot` is described in [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025).  |


