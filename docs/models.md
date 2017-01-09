When calling the training script you can specify a model class using the `--model` flags and model-specific hyperparameters using the `--hparams` flag. This page list all support models and hyperparameters.

## Common Hyperparameters

The following hyperparameters are sahred by all models, unless explicitly stated in the model section.

| Name | Default | Description |
| --- | --- | --- |
| `source.max_seq_len` | `40` | Maximum length of source sequences. An example is sliced to this length before being fed to the encoder. |
| `source.reverse` | `True` | If set to true, reverse the source sequence before feeding it into the encoder.|
| `target.max_seq_len` | `40` | Maximum length of target sequences. An example is sliced to this length before being fed to the decoder. |
| `embedding.dim` | `100` | Dimensionality of the embedding layer. |
| `optimizer.name` | `Adam` | Type of Optimizer to use, e.g. `Adam`, `SGD` or `Momentum`. The name is fed to Tensorflow's [optimize_loss](https://www.tensorflow.org/api_docs/python/contrib.layers/optimization#optimize_loss) function. See Tensorflow documentation for more details and all available options. |
| `optimizer.learning_rate` | `1e-4` | Initial learning rate for the optimizer. This is fed to Tensorflow's [optimize_loss](https://www.tensorflow.org/api_docs/python/contrib.layers/optimization#optimize_loss) function. |
| `optimizer.lr_decay_type` |  | The name of one of Tensorflow's [learning rate decay function](https://www.tensorflow.org/api_docs/python/#training--decaying-the-learning-rate) defined in `tf.train`, e.g. `exponential_decay`. If this is an empty string (default) then no learning rate decay is used. |
| `optimizer.lr_decay_steps` | `100` | How often to apply decay. This is fed as the `decay_steps` argument to the decay function defined above. See Tensoflow documentation for more details. |
| `optimizer.lr_decay_rate` | `0.99` | The decay rate. This is fed as the `decay_rate` argument to the decay function defined above. See Tensorfow documentation for more details. |
| `optimizer.lr_start_decay_at` | `0` | Start learning rate decay at this step. |
| `optimizer.lr_stop_decay_at` | `1e9` | Stop learning rate decay at this step.  |
| `optimizer.lr_min_learning_rate` | `1e-12` | Never decay below this learning rate. |
| `optimizer.lr_staircase` | `False` | If `True` decay the learning rate at discrete intervals. This is fed as the `staircase` argument to the decay function defined above. See Tensorfow documentation for more details. |
| `optimizer.clip_gradients` | `5.0` | Clip gradients by their global norm. |

## BasicSeq2Seq

The `BasicSeq2Seq` model uses a unidirectional encoder and decoder without attention mechanism. The last encoder state is used to initialize the decoder. This is the "vanilla" implementation of the seq2seq architecture.

This model suports the following additional hyperparameters.


| Name | Default | Description |
| --- | --- | --- |
| `rnn_cell.type` | `BasicLSTMCell` | The class name of a RNN Cell defined in `tf.contrib.rnn`. The most common values are `BasicLSTMCell`, `LSTMCell` or `GRUCell`. Applies to both encoder and decoder. |
| `rnn_cell.num_units` | `128` | The number of units to use for the RNN Cell. Applies to both encoder and decoder. |
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. Applies to both encoder and decoder. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. Applies to both encoder and decoder. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. Applies to both encoder and decoder. |



## AttentionSeq2seq

`AttentionSeq2seq` is a sequence to sequence model that uses a unidirectional or bidirectional encoder and a decoder with attention mechanism. The last encoder state is not fed to the decoder. This implementation is comparable to the model in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

| Name | Default | Description |
| --- | --- | --- |
| `attention.dim` | `128` | Number of units in the attention layer. |
| `attention.score_type` | `dot` | The formula used to calculate attention scores. Available values are `bahdanau` and `dot`. `bahdanau` is described in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). `dot` is described in [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025).  |
| `encoder.type` | `UnidirectionalRNNEncoder` | Type of encoder to use. This is the class name of an encoder defined in `seq2seq.encoder`. Currently the supported value are `BidirectionalRNNEncoder` and `UnidirectionalRNNEncoder`. |
| `encoder.rnn_cell.type` | `BasicLSTMCell` | The class name of a RNN Cell defined in `tf.contrib.rnn`. The most common values are `BasicLSTMCell`, `LSTMCell` or `GRUCell`. |
| `encoder.rnn_cell.num_units` | `128` | The number of units to use for the RNN Cell. |
| `encoder.rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `encoder.rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `encoder.rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `decoder.rnn_cell.type` | `BasicLSTMCell` | The class name of a RNN Cell defined in `tf.contrib.rnn`. The most common values are `BasicLSTMCell`, `LSTMCell` or `GRUCell`. |
| `decoder.rnn_cell.num_units` | `128` | The number of units to use for the RNN Cell. |
| `decoder.rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `decoder.rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `decoder.rnn_cell.num_layers` | `1` | Number of RNN layers. |

