## Introduction

tf-seq2seq is an open source framework for sequence learning in Tensorflow.
The main use case is Neural Machine Translation (NMT), but tf-seq2seq can be
used for a variety of other applications such as Conversational Modeling,
Text Summarization or Image Captioning.

## Design Goals

We built tf-seq2seq with the following goals in mind:

1. **Usability**. You can train a model with a single command. The input data are stored in raw text rather than obscure file formats.

2. **Extensibility**. Code is structured so that it is easy to build upon. For example, adding a new attention mechanism or encoder architecture requires only minimal code changes.

3. **Full Documentation**. All code is documented using standard Python docstrings, and we have written guides to help you get started with tf-seq2seq.

4. **Good Performance**. For the sake of code simplicity, we do not try to squeeze out every last bit of performance, but the implementation is fast enough to cover almost all production and research use cases. tf-seq2seq also supports distributed training to trade off computational power and training time.

5. **Standard Benchmarks**. We provide [configuration files and benchmark results](benchmarks.md) for standard datasets and models. We hope these can serve as a baseline for future research.

## Related Frameworks

The following frameworks offer functionality similar to that of tf-seq2seq. We hope to collaborate with the authors of these frameworks so that we can learn from each other.

- [OpenNMT (Torch)](http://opennmt.net/)
- [Neural Monkey (Tensorflow)](https://github.com/ufal/neuralmonkey)
- [NEMATUS (Theano)](https://github.com/rsennrich/nematus)
