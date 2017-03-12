## Encoder Reference

All encoders inherit from the abstract `Encoder` defined in `seq2seq.encoders.encoder` and receive `params`, `mode` arguments at instantiation time. Available hyperparameters vary by encoder class.

### [`UnidirectionalRNNEncoder`](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/rnn_encoder.py)

---

| Name | Default | Description |
| --- | --- | --- |
| `rnn_cell.cell_class` | `BasicLSTMCell` | The class of the rnn cell. Cell classes can be fully defined (e.g. `tensorflow.contrib.rnn.BasicRNNCell`) or must be in `tf.contrib.rnn` or `seq2seq.contrib.rnn_cell`. |
| `rnn_cell.cell_params` | `{"num_units": 128}` | A dictionary of parameters to pass to the cell class constructor. |
| `rnn_cell.dropout_input_keep_prob` | `1.0` | Apply dropout to the (non-recurrent) inputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.dropout_output_keep_prob` | `1.0`| Apply dropout to the (non-recurrent) outputs of each RNN layer using this keep probability. A value of `1.0` disables dropout. |
| `rnn_cell.num_layers` | `1` | Number of RNN layers. |
| `rnn_cell.residual_connections` | `False` | If true, add residual connections between RNN layers in the encoder. |

### [`BidirectionalRNNEncoder`](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/rnn_encoder.py)

---

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.

### [`StackBidirectionalRNNEncoder`](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/rnn_encoder.py)

---

Same as the `UnidirectionalRNNEncoder`. The same cell is used for forward and backward RNNs.


### [`PoolingEncoder`](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/pooling_encoder.py)

---

An encoder that pools over embeddings, as described in [https://arxiv.org/abs/1611.02344](https://arxiv.org/abs/1611.02344). The encoder supports optional positions embeddings and a configurable pooling window.


| Name | Default | Description |
| --- | --- | --- |
| `pooling_fn` | `tensorflow.layers.average_pooling1d` | The 1-d pooling function to use, e.g. `tensorflow.layers.average_pooling1d`. |
| `pool_size` | `5` | The pooling window, passed as `pool_size` to the pooling function. |
| `strides` | `1` | The stride during pooling, passed as `strides` the pooling function. |
| `position_embeddings.enable` | `True` | If true, add position embeddings to the inputs before pooling. |
| `position_embeddings.combiner_fn` | `tensorflow.add` | Function used to combine the position embeddings with the inputs. For example, `tensorflow.add`. |
| `position_embeddings.num_positions` | `100` | Size of the position embedding matrix. This should be set to the maximum sequence length of the inputs. |


### [`InceptionV3Encoder`](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/image_encoder.py)

---

**This encoder is experimental**. This encoder puts the image through an InceptionV3 network and uses the last
hidden layer before the logits as the feature representation.

| Name | Default | Description |
| --- | --- | --- |
| `resize_height` | `299` | Resize the image to this height before feeding it into the convolutional network. |
| `resize_width` | `299` | Resize the image to this width before feeding it into the convolutional network. |


