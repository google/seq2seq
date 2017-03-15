## Introduction

tf-seq2seq is a general-purpose encoder-decoder framework for Tensorflow that can be used for Machine Translation, Text Summarization, Conversational Modeling, Image Captioning, and more.

![Machine Translation Model](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)

## Design Goals

We built tf-seq2seq with the following goals in mind:

- **General Purpose**: We initially built this framework for Machine Translation, but have since used it for a variety of other tasks, including Summarization, Conversational Modeling, and Image Captioning. As long as your problem can be phrased as encoding input data in one format and decoding it into another format, you should be able to use or extend this framework.

- **Usability**: You can train a model with a single command. Several types of input data are supported, including standard raw text.

- **Reproducibility**: Training pipelines and models are configured using YAML files. This allows other to run your exact same model configurations.

- **Extensibility**: Code is structured in a modular way and that easy to build upon. For example, adding a new type of attention mechanism or encoder architecture requires only minimal code changes.

- **Documentation**: All code is documented using standard Python docstrings, and we have written guides to help you get started with common tasks.

- **Good Performance**: For the sake of code simplicity, we did not try to squeeze out every last bit of performance, but the implementation is fast enough to cover almost all production and research use cases. tf-seq2seq also supports distributed training to trade off computational power and training time.


## FAQ

**1. How does this framework compare to the [Google Neural Machine Translation](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html) system? Is this the official open-source implementation?**

No, this is not an official open-source implementation of the GNMT system. This framework was built from the bottom up to cover a wider range of tasks, Neural Machine Translation being one of them. We have not replicated the exact GNMT architecture in this framework, but we welcome [contributions](contributing.md) in that direction.


## Related Frameworks

The following frameworks offer functionality similar to that of tf-seq2seq. We hope to collaborate with the authors of these frameworks so that we can learn from each other.

- [OpenNMT (Torch)](http://opennmt.net/)
- [Neural Monkey (Tensorflow)](https://github.com/ufal/neuralmonkey)
- [NEMATUS (Theano)](https://github.com/rsennrich/nematus)
