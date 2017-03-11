When calling the training script, you can specify a model class using the `--model` flag and model-specific hyperparameters using the `--model_params` flag. This page lists all supported models and hyperparameters.

## [`ModelBase`](https://github.com/google/seq2seq/blob/master/seq2seq/models/model_base.py)
---

This is an abstract class that cannot be used as a model during training. Other model classes inherit from this. The following parameters are shared by all models, unless explicitly stated otherwise in the model section.

| Name | Default | Description |
| --- | --- | --- |
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


## [`Seq2SeqModel`](https://github.com/google/seq2seq/blob/master/seq2seq/models/seq2seq_model.py)
---

This is an abstract class that cannot be used as a model during training. Other model classes inherit from this. The following hyperparameters are shared by all models that inherit from `Seq2SeqModel`, unless explicitly stated otherwise.

| Name | Default | Description |
| --- | --- | --- |
| `source.max_seq_len` | `50` | Maximum length of source sequences. An example is sliced to this length before being fed to the encoder. |
| `source.reverse` | `True` | If set to true, reverse the source sequence before feeding it into the encoder.|
| `target.max_seq_len` | `50` | Maximum length of target sequences. An example is sliced to this length before being fed to the decoder. |
| `embedding.dim` | `100` | Dimensionality of the embedding layer. |
| `embedding.share` | `False` | If set to true, share embedding parameters for source and target sequences. |
| `inference.beam_search.beam_width` | `0` | Beam Search beam width used during inference. A value of less or equal than `1` disables beam search. |
| `inference.max_decode_length` | `100` | During inference mode, decode up to this length or until a `SEQUENCE_END` token is encountered, whichever happens first. |
| `inference.beam_search.length_penalty_weight` | `0.0` | Length penalty factor applied to beam search hypotheses, as described in [https://arxiv.org/abs/1609.08144](https://arxiv.org/abs/1609.08144). |
| `vocab_source` | `""` | Path to the source vocabulary to use. This is used to map input tokens to integer IDs. |
| `vocab_target` | `""` | Path to the target vocabulary to use. This is used to map input tokens to integer IDs. |

## [`BasicSeq2Seq`](https://github.com/google/seq2seq/blob/master/seq2seq/models/basic_seq2seq.py)
---

Includes all parameters from `Seq2SeqModel`. The `BasicSeq2Seq` model uses an encoder and decoder with no attention mechanism. The last encoder state is passed through a fully connected layer and used to initialize the decoder (this behavior can be changed using the `bridge.*` hyperparameter). This is the "vanilla" implementation of the standard seq2seq architecture.

| Name | Default | Description |
| --- | --- | --- |
| `bridge.class` | `seq2seq.models.bridges.InitialStateBridge` | Type of bridge to use. The bridge defines how state is passed between the encoder and decoder. Refer to the [`seq2seq.models.bridges`](https://github.com/google/seq2seq/blob/master/seq2seq/models/bridges.py) module for more details. |
| `bridge.params` | `{}` | Parameters passed to the bridge during construction. |
| `encoder.class` | `seq2seq.encoders.UnidirectionalRNNEncoder` | Type of encoder to use. See the [Encoder Reference](encoders/) for more details and available encoders. |
| `encoder.params` | `{}` | Parameters passed to the encoder during construction. See the [Encoder Reference](encoders/) for more details.|
| `decoder.class` | `seq2seq.decoders.BasicDecoder` | Type of decoder to use. See the [Decoder Reference](decoders/) for more details and available encoders. |
| `decoder.params` | `{}` | Parameters passed to the decoder during construction. See the [Decoder Reference](decoders/) for more details.|


## [`AttentionSeq2Seq`](https://github.com/google/seq2seq/blob/master/seq2seq/models/attention_seq2seq.py)
---

Includes all parameters from `Seq2SeqModel` and `BasicSeq2Seq`. This model is similar to `BasicSeq2Seq`, except that it uses an attention mechanism during decoding. By default, the last encoder state is not fed to the decoder.  The implementation is comparable to the model in [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

| Name | Default | Description |
| --- | --- | --- |
| `attention.class` | `AttentionLayerBahdanau` | Class name of the attention layer. Can be a fully-qualified name or is assumed to be defined in `seq2seq.decoders.attention`. Currently available layers are `AttentionLayerBahdanau` and `AttentionLayerDot`. |
| `attention.params` | `{"num_units": 128}` | A dictionary of  parameters passed to the attention class constructor. |
| `bridge.class` | `seq2seq.models.bridges.ZeroBridge` | Type of bridge to use. The bridge defines how state is passed between the encoder and decoder. Refer to the [`seq2seq.models.bridges`](https://github.com/google/seq2seq/blob/master/seq2seq/models/bridges.py) module for more details. |
| `encoder.class` | `seq2seq.encoders.BidirectionalRNNEncoder` | Type of encoder to use. See the [Encoder Reference](encoders/) for more details and available encoders. |
| `decoder.class` | `seq2seq.decoders.AttentionDecoder` | Type of decoder to use. See the [Decoder Reference](decoders/) for more details and available encoders. |


## [`Image2Seq`](https://github.com/google/seq2seq/blob/master/seq2seq/models/image2seq.py)
---

**This model is currently experimental.** This model uses the same parameters as `AttentionSeq2Seq` with the following differences:

- The default encoder is `seq2seq.encoders.InceptionV3Encoder`
- There are not `source.max_seq_len` and `source.reverse`, and `vocab_source` parameters.