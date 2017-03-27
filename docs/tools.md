## Generating Vocabulary

A vocabulary file is a raw text file that contains one word per line, followed by a tab separator and the word count. The total number of lines is equal to the size of the vocabulary and each token is mapped to its line number. We provide a helper script [`bin/tools/generate_vocab.py`](https://github.com/google/seq2seq/blob/master/bin/tools/generate_vocab.py) that takes in a raw text file of space-delimited tokens and generates a vocabulary file:

```shell
./bin/tools/generate_vocab.py < data.txt > vocab
```


## Generating Character Vocabulary

Sometimes you want to run training on characters instead of words or subword units. Using the same script [`bin/tools/generate_vocab.py`](https://github.com/google/seq2seq/blob/master/bin/tools/generate_vocab.py) with `--delimiter ""` can generate a vocabulary file that contains the unique set of characters found in the text:

```shell
./bin/tools/generate_vocab.py --delimiter "" < data.txt > vocab
```

To run training on characters you must pass set `source_delimiter` and `target_delimiter` delimiter of the input pipeline to `""`. See the [Training documentation](training.md) for more details.


## Visualizing Beam Search

If you use the `DumpBeams` inference task (see [Inference](inference/) for more details) you can inspect the beam search data by loading the array using numpy, or generate beam search visualizations using the `generate_beam_viz.py` script. This required the `networkx` module to be installed.

```
python -m bin.tools.generate_beam_viz  \
  -o ${TMPDIR:-/tmp}/beam_visualizations \
  -d ${TMPDIR:-/tmp}/beams.npz \
  -v $HOME/nmt_data/toy_reverse//train/vocab.targets.txt
```

![Beam Search Visualization](http://i.imgur.com/kLec8l4l.png)
