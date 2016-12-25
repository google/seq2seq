## Introduction

[seq2seq] is an open source framework for sequence learning in Tensorflow.
The main use case is Neural Machine Translation (NMT), but [seq2seq] can be
used for a variety of other applications such as Conversational Modeling,
Text Summarization or Image Captioning.

## Design Goals

We built [seq2seq] with the following goals in mind.

1. **Ease of use.** You can train a model with single command. The input data are raw text files instead of esoteric file formats. Using  pre-trained models to make predictions should be straightforward.

2. **Easy to extend**. Code is structured so that it is easy to build upon. For example, adding a new type of attention mechanism or a new encoder architecture requires only minimal code changes.

3. **Well-documented**. In addition the [API documentation]() we have written up guides to help you become familiar with [seq2seq].

4. **Good performance**. For the sake of code simplicity we did not triy to squeeze out every last bit of performance, but the implementation is fast enough to cover most production use cases. [seq2seq] also supports distributed training to trade off computational power and training time.

5. **Standard Benchmarks**. We provide [pre-trained models and benchmark results](benchmarks.md) for several standard datasets. We hope these can serve as a baseline for future research.

## Related Frameworks

- [OpenNMT (Torch)](http://opennmt.net/)
- [Stanford NMT (Matlab)](http://nlp.stanford.edu/projects/nmt/)
- [Neural Monkey (Tensorflow)](https://github.com/ufal/neuralmonkey)
- [NEMATUS (Theano)](https://github.com/rsennrich/nematus)
