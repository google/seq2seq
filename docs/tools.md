## Visualizing Attention

If you trained an `AttentionSeq2Seq` model you can use the `bin/print_attention.py` script to dump the raw attention scores and generate alignment visualizations. The usage is similar to the inference script and uses the same input data. For example:

```
./bin/print_attention.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.sources.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.targets.txt \
  --model AttentionSeq2Seq \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --output_dir ${TMPDIR}/attention_plots
```

By default, the script generates an `attention_score.npy` array file and one attention plot per example. The array file can be [loaded used numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html) and will contain a list of arrays with shape `[target_length, source_length]`. If you only want the raw attention score data without the plots you can pass the `--no_plot` flag. For more details and additional options see the [`print_attention.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/print_attention.py) file.


## Visualizing Beam Search

Not yet supported.


## Model Performance Profiling

During training the [MetadataCaptureHook](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/training/hooks.py) saves a full  trace and timeline information for a single training step (step 10 by default) into a
a `metadata` subdirectory of your model directory. You can view the generated `timeline.json` file in Chrome:

1. Go to `chrome://tracing`
2. Load the `timeline.json` file that was saved in `/path/to/model/dir/metadata`

For large complicated graphs the timeline files can become quite large and analyzing them using Chrome may be slow, which is why we also provide a [`profile.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/tools/profile.py) script tthat generates useful information:

```shell
./bin/tools/profile.py --model_dir=/path/to/model/dir
```

This command will generate 4 files:

- `/path/to/model/dir/params.txt` contains an analysis of model parameters, including the number of parameters and their shapes and sizes
- `/path/to/model/dir/flops.txt` contains information about long-running floating point operations per second (FLOPS)
- `/path/to/model/dir/micro.txt` contains microsecond timing information for operations that take longer than 1 millisecond, organized by graph structure
- `/path/to/model/dir/device.txt` contains detailed device placement information for all operations
