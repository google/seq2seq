## Download & Setup

To use tf-seq2seq you need a working installation of TensorFlow 1.0 with
Python 2.7 or Python 3.5. Follow the [TensorFlow Getting Started](https://www.tensorflow.org/versions/r1.0/get_started/os_setup) guide for detailed setup instructions. With TensorFlow installed, you can clone this repository:

```bash
git clone https://github.com/google/seq2seq.git
cd seq2seq

# Install package and dependencies
pip install -e .
```

To make sure everything works as expect you can run a simple pipeline unit test:

```bash
python -m unittest seq2seq.test.pipeline_test
```

If you see a "OK" message, you are all set. Note that you may need to install pyrouge, pyyaml, and matplotlib, in order for these tests to pass. If you run into other setup issues,
please [file a Github issue](https://github.com/google/seq2seq/issues).

## Common Installation Issues

### Incorrect matploblib backend

In order to generate plots using matplotlib you need to have set the correct [backend](http://matplotlib.org/faq/usage_faq.html#what-is-a-backend). Also see this [StackOverflow thread](http://stackoverflow.com/questions/4930524/how-can-i-set-the-backend-in-matplotlib-in-python). To use the `Agg` backend, simply:

```
echo "backend : Agg" >> $HOME/.config/matplotlib/matplotlibrc
```

## Next Steps

- Learn about [concepts and terminology](concepts.md)
- Read through the [Neural Machine Translation Tutorial](nmt.md)
- Use [pre-processed datasets](data.md) or train a model on your own data
- [Contribute!](contributing.md)
