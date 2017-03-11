## Configuration

Many objects, including Encoders, Decoders, Models, Input Pipelines, and Inference Tasks, are configured using key-value parameters. These parameters are typically passed as [YAML](https://en.wikipedia.org/wiki/YAML) through configuration files or directly on the command line. For example, you can pass a `model_params` string to the training script configure model. Configurations are often be nested, as in the following example:

```yml
model_params:
  attention.class: seq2seq.decoders.attention.AttentionLayerBahdanau
  attention.params:
    num_units: 512
  embedding.dim: 1024
  encoder.class: seq2seq.encoders.BidirectionalRNNEncoder
  encoder.params:
    rnn_cell:
      cell_class: LSTMCell
      cell_params:
        num_units: 512
```

## Input Pipeline

An [`InputPipeline`](https://github.com/google/seq2seq/blob/master/seq2seq/data/input_pipeline.py) defines how data is read, parsed, and separated into features and labels. For example, the `ParallelTextInputPipeline` reads data from two text files, separates tokens by a delimiter, and produces tensors corresponding to the `source_tokens`, `source_length`, `target_tokens`, and `target_length` for each example. If you want to read new data formats you need to implement your own input pipeline.

## Encoder

An encoder reads in "source data", e.g. a sequence of words or an image, and produces a feature representation in continuous space. For example, a Recurrent Neural Network encoder may take as input a sequence of words and produce a fixed-length vector that roughly corresponds to the meaning of the text. An encoder based on a Convolutional Neural Network may take as input an image and generate a new volume that contains higher-level features of the image. The idea is that the representation produced by the encoder can be used by the Decoder to generate new data, e.g. a sentence in another language, or the description of the image. For a list of available encoders, see the [Encoder Reference](encoders/).


## Decoder

A decoder is a generative model that is conditioned on the representation created by the encoder. For example, a Recurrent Neural Network decoder may learn generate the translation for an encoded sentence in another language. For a list of available decoder, see the [Decoder Reference](decoders/).


## Model

A model defines how to put together an encoder and decoder, and how to calculate and minize the loss functions. It also handles the necessary preprocessing of data read from an input pipeline. Under the hood, each model is implemented as a [model_fn passed to a tf.contrib.learn Estimator](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Estimator). For a list of available models, see the [Models Reference](models/).

