## Inference Tasks

When calling the inference script `bin/infer.py`, you must provide a list of tasks to run. The most basic task, `DecodeText`, simply prints out the model predictions. By additing more tasks you can perform additional features, such as storing debugging infromation or visualization attention scores. Under the hood, each `InferenceTask` is implemented as a Tensorflow [SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) that requests outputs from the model and knows how to process them.

## DecodeText

The `DecodeText` task reads the model predictions and prints the predictions to standard output. It has the following parameters:

- `delimiter`: String to join the tokens predicted by the model on. Defaults to space.
- `unk_replace`: If set to `True`, perform unknown token replacement based on attention scores. Default is `False`. See below for more details.
- `unk_mapping`: If set to the path of a dictionary file, use the provided mapping to perform unknown token replacement. See below for more details.

#### UNK token replacement using a Copy Mechanism

Rare words (such as place and people names) are often absent from the target vocabulary and result in `UNK` tokens in the output predictions. An easy strategy to target sequences is to replace each `UNK` token with the word in the source sequence it is best aligned with. Alignments are typically calculated using an attention mechanism which produces alignment scores for each target token. If you trained a model that generates such attention scores (e.g. `AttentionSeq2Seq`), you can use them to perform UNK token replacement by activating the `unk_replace` parameter.


```bash
mkdir -p ${DATA_PATH}/pred
python -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True"
```

#### UNK token replacement using a mapping

A more sophisticated approach to UNK token replacement is to use a mapping instead of copying words from the source. For example, the English word "Munich" is usually translated as "München" in German. Simply copying "Munich" from the source you would never result in the right translation even if the words were perfectly aligned using attention scores. One strategy is to use [fast_align](https://github.com/clab/fast_align) to generate a mapping based on the conditional probabilities of target given source.

```bash
# Download and build fast_align
git clone https://github.com/clab/fast_align.git
mkdir fast_align/build && cd fast_align/build
cmake ../ && make

# Convert your data into a format that fast_align understands:
# <source> ||| <target>
paste \
  $HOME/nmt_data/toy_reverse/train/sources.txt \
  $HOME/nmt_data/toy_reverse/train/targets.txt \
  | sed "s/$(printf '\t')/ ||| /g" > $HOME/nmt_data/toy_reverse/train/source_targets.fastalign

# Learn alignments
./fast_align \
  -i $HOME/nmt_data/toy_reverse/train/source_targets.fastalign \
  -v -p $HOME/nmt_data/toy_reverse/train/source_targets.cond \
  > $HOME/nmt_data/toy_reverse/train/source_targets.align

# Find the most probable traslation for each word and write them to a file
sort -k1,1 -k3,3gr $HOME/nmt_data/toy_reverse/train/source_targets.cond \
  | sort -k1,1 -u \
  > $HOME/nmt_data/toy_reverse/train/source_targets.cond.dict

```

The output file specified by the `-p` argument will contain conditional probabilities for `p(target | source)` in the form of `<source>\t<target>\t<prob>`. These can be used to do smarter UNK token replacement by passing the `unk_mapping` flag.

```bash
mkdir -p ${DATA_PATH}/pred
python -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True"
        unk_mapping: $HOME/nmt_data/toy_reverse/train/source_targets.cond.dict"
  ...
```


## Visualizing Attention

If you trained a model using the  `AttentionDecoder`, you can dump the raw attention scores and generate alignment visualizations during inference using the `DumpAttention` task.

```shell
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpAttention
      params:
        output_dir: $HOME/attention" \
  ...
```

By default, this script generates an `attention_score.npy` array file and one attention plot per example. The array file can be [loaded used numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html) and will contain a list of arrays with shape `[target_length, source_length]`. If you only want the raw attention score data without the plots you can enable the `dump_atention_no_plot` parameter.



## Dumping Beams

If you are using beam search during decoding, you can use the `DumpBeams` task to write beam search debugging information to disk. You can later inspect the data using numpy, or use the [provided script](tools/) to generate visualizations.

```shell
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${TMPDIR:-/tmp}/wmt_16_en_de/newstest2014.pred.beams.npz" \
  --model_params "
    inference.beam_search.beam_width: 5" \
  ...
```
