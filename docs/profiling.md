There are various ways to profile your model, from printing out the number of parameters to viewing detailed timing and device placement information for all graph operations.

### `MetadataCaptureHook`

The easiest way to generate profiling information is to use the [MetadataCaptureHook](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/training/hooks.py#L16), which saves a full graph trace and timeline information for a single configurable training step. By default these files are saved a `metadata` subdirectory of your model checkpoint directory.

### Timelines

If you used the `MetadataCaptureHook` you can view the generated `timeline.json` file in your web browser:

1. Go to `chrome://tracing`
2. Load the `timeline.json` file that was saved by the `MetadataCaptureHook`.

For complicated graphs timeline files can become quite large and analyzing them using Chrome may be slow and inconvenient. 

### Profiling Script

An easy way to get basic information about your model is to run the [`profile.py`](https://github.com/dennybritz/seq2seq/blob/master/bin/tools/profile.py) script. It uses [TFProf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tfprof) to read metadata saved by the `MetadataCaptureHook` and generates several analyses.

```shell
./bin/tools/profile.py --model_dir=/path/to/model/dir
```

This command will generate 4 files:

`/path/to/model/dir/params.txt` contains an analysis of model parameter, including the number of parameters and their shapes and sizes:

```
att_seq2seq/attention_decoder/RNN/rnn_step/logits/weights (384x103, 39.55k/39.55k params, 158.21KB/158.21KB)
```

`/path/to/model/dir/flops.txt` contains information about floating point operations:

```
att_seq2seq/OptimizeLoss/gradients/att_seq2seq/attention_decoder/RNN/while/rnn_step/attention/inputs_att/MatMul_grad/MatMul (18.87m/18.87m flops, 1.64ms/1.64ms, /job:localhost/replica:0/task:0/cpu:0)
```

`/path/to/model/dir/micro.txt` contains microsecond timing information for operations that take longer than 1 milliseconds, organized by graph structure:

```
att_seq2seq/attention_decoder/RNN/while/rnn_step/attention/mul_1 (1.89ms/13.72ms, /job:localhost/replica:0/task:0/cpu:0)
  att_seq2seq/attention_decoder/RNN/while/rnn_step/attention/inputs_att/MatMul (1.21ms/11.83ms, /job:localhost/replica:0/task:0/cpu:0)
  ....
```

`/path/to/model/dir/device.txt` contains detailed device placement information for all operations.






