## Download & Setup

To use tf-seq2seq you need a working installation of TensorFlow 1.0 with
Python 2.7 or Python 3.5. Follow the [TensorFlow Getting Started](https://www.tensorflow.org/versions/r1.0/get_started/os_setup) guide for detailed setup instructions. With TensorFlow installed, you can clone this repository:

```bash
git clone https://github.com/google/seq2seq.git
cd seq2seq
```

To make sure everything works as expect you can run a simple pipeline unit test:

```bash
python -m unittest seq2seq.test.pipeline_test
```

If you see a "success" message, you are all set. Note that you may need to install pyrouge, pyyaml, and matplotlib, in order for these tests to pass. If you run into other setup issues,
please [file a Github issue](https://github.com/google/seq2seq/issues).


## Next Steps

- Learn about [concepts and terminology](concepts/)
- Read through the [Neural Machine Translation Tutorial](nmt/)
- Use [pre-processed datasets](data/) or train a model on your own data
- [Contribute!](contributing)