## Visualizing Attention

If you train an `AttentionSeq2Seq` model, you can dump the raw attention scores and generate alignment visualizations during inference:

```
./bin/infer.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --dump_atention_dir ${TMPDIR}/attention_plots > /dev/null
```

By default, this script generates an `attention_score.npy` array file and one attention plot per example. The array file can be [loaded used numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html) and will contain a list of arrays with shape `[target_length, source_length]`. If you only want the raw attention score data without the plots you can pass the `--dump_atention_no_plot` flag. For more details and additional options, see the [`infer.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/infer.py) script.


## Visualizing Beam Search

To dump beam search debugging information you can pass the `--dump_beams` flag to the `infer.py script`. This will
write a numpy npz with the raw beam search data.


```
./bin/infer.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --beam_width 5 \
  --dump_beams ${TMPDIR}/beams.npz > /dev/null
```

You can inspect the beam search data by loading the array using numpy, or generate beam search visualizations using the
`generate_beam_viz.py` script:

```
bin/tools/generate_beam_viz.py  \
  -o ${TMPDIR}/beam_visualizations \
  -d ${TMPDIR}/beams.npz \
  -v $HOME/nmt_data/toy_reverse//train/vocab.targets.txt
```

![Beam Search Visualization](http://i.imgur.com/kLec8l4l.png)


## Model Performance Profiling

During training, the [MetadataCaptureHook](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/training/hooks.py) saves a full trace and timeline information for a single training step (step 10 by default) into a `metadata` subdirectory of your model directory. You can view the generated `timeline.json` file in Chrome:

1. Go to `chrome://tracing`
2. Load the `timeline.json` file that was saved in `/path/to/model/dir/metadata`

For large complicated graphs, the timeline files can become quite large and analyzing them using Chrome may be slow, which is why we also provide a [`profile.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/tools/profile.py) script that generates useful profiling information:

```shell
./bin/tools/profile.py --model_dir=/path/to/model/dir
```

This command will generate the following four files:

- `/path/to/model/dir/params.txt` contains an analysis of model parameters, including the number of parameters and their shapes and sizes
- `/path/to/model/dir/flops.txt` contains information about long-running floating point operations per second (FLOPS)
- `/path/to/model/dir/micro.txt` contains microsecond timing information for operations that take longer than 1 millisecond, organized by graph structure
- `/path/to/model/dir/device.txt` contains detailed device placement information for all operations
