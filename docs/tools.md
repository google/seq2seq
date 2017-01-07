## Tokenization

TODO

## Vocabulary Generation

TODO

## Debugging Attention

If you trained an `AttentionSeq2Seq` model you can use the `bin/print_attention.py` script to dump the raw attention scores and generate alignment visualizations. The usage is similar to the inference script and uses the same input data. For example:

```
./bin/print_attention.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.txt \
  --model AttentionSeq2Seq \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --output_dir ${TMPDIR}/attention
```

By default, the script generates an `attention_score.npy` array file and an attention plot per example. The array file can be [loaded used numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html). It will be of shape `[num_examples, target_length, source_length]`. If you want only the raw attention score data without the plots you can pass the `--no_plot` flag. For more details and additional options see the [`print_attention.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/print_attention.py) file.


## Using Subword Units (BPE)

TODO

## Using an Alignment Dictionary

TODO


## Visualizing Beam Search

TODO

