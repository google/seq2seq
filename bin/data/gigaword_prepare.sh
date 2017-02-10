#! /usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

DATA_DIR=${DATA_DIR:-$HOME/nmt_data}
echo "Writing to ${DATA_DIR}. To change this, set the DATA_DIR environment variable."

OUTPUT_DIR=${DATA_DIR}/gigaword
mkdir -p $OUTPUT_DIR

mkdir -p $OUTPUT_DIR/extracted

# Unzip all files
find  $GIGAWORD_DIR -name *.gz | xargs gunzip

# Process all data files
for f in $(find $GIGAWORD_DIR -type f | grep data/); do
  outfile="$OUTPUT_DIR/extracted/$(basename $f)"
  $BASE_DIR/bin/data/gigaword_extract.py -f $f -o $outfile
  echo "$(wc -l ${outfile}/sources.txt)"
  echo "$(wc -l ${outfile}/targets.txt)"
done

# Combine all sources and targets
find $OUTPUT_DIR/extracted -name sources.txt | sort | xargs cat > $OUTPUT_DIR/combined.sources
find $OUTPUT_DIR/extracted -name targets.txt | sort | xargs cat > $OUTPUT_DIR/combined.targets

echo "Found $(wc -l $OUTPUT_DIR/combined.sources) sources."
echo "Found $(wc -l $OUTPUT_DIR/combined.targets) targets."

# Get moses
if [ ! -d "${DATA_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${DATA_DIR}/mosesdecoder"
fi

# Clean data
${DATA_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl \
  --ignore-ratio \
  ${OUTPUT_DIR}/combined sources targets "${OUTPUT_DIR}/combined.clean" 5 200

# Lowercase data
# A lot of the headlines in Gigaword are all caps
tr '[:upper:]' '[:lower:]' < ${OUTPUT_DIR}/combined.clean.sources > ${OUTPUT_DIR}/combined.lc.sources
tr '[:upper:]' '[:lower:]' < ${OUTPUT_DIR}/combined.clean.targets > ${OUTPUT_DIR}/combined.lc.targets

# Tokenize data
${DATA_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl \
  -q -l en -threads 8 \
  < ${OUTPUT_DIR}/combined.lc.sources \
  > ${OUTPUT_DIR}/combined.tok.sources

${DATA_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl \
  -q -l en -threads 8 \
  < ${OUTPUT_DIR}/combined.lc.targets \
  > ${OUTPUT_DIR}/combined.tok.targets


# Shuffle data in parallel
# Source: http://unix.stackexchange.com/questions/220390/shuffle-two-parallel-text-files?
mkfifo  ${OUTPUT_DIR}/random1 ${OUTPUT_DIR}/random2
tee ${OUTPUT_DIR}/random1 ${OUTPUT_DIR}/random2 < /dev/urandom > /dev/null &
shuf --random-source=${OUTPUT_DIR}/random1 \
  ${OUTPUT_DIR}/combined.tok.sources > ${OUTPUT_DIR}/combined.tok.shuf.sources &
shuf --random-source ${OUTPUT_DIR}/random2 \
  ${OUTPUT_DIR}/combined.tok.targets > ${OUTPUT_DIR}/combined.tok.shuf.targets &
wait
rm ${OUTPUT_DIR}/random1 ${OUTPUT_DIR}/random2


# Split into train/dev/test
tail -n +20000 ${OUTPUT_DIR}/combined.tok.shuf.sources > ${OUTPUT_DIR}/train.sources
tail -n +20000 ${OUTPUT_DIR}/combined.tok.shuf.targets > ${OUTPUT_DIR}/train.targets
head -n 10000 ${OUTPUT_DIR}/combined.tok.shuf.sources > ${OUTPUT_DIR}/dev.sources
head -n 10000 ${OUTPUT_DIR}/combined.tok.shuf.targets > ${OUTPUT_DIR}/dev.targets
head -n 20000 ${OUTPUT_DIR}/combined.tok.shuf.sources | tail -n 10000 > ${OUTPUT_DIR}/test.sources
head -n 20000 ${OUTPUT_DIR}/combined.tok.shuf.targets | tail -n 10000 > ${OUTPUT_DIR}/test.targets

# Learn BPE
if [ ! -d "${DATA_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${DATA_DIR}/subword-nmt"
fi

BPE_MERGE_OPS=32000
echo "Learning BPE with merge_ops=${BPE_MERGE_OPS}. This may take a while..."
cat "${OUTPUT_DIR}/train.sources" "${OUTPUT_DIR}/train.targets" | \
    ${DATA_DIR}/subword-nmt/learn_bpe.py -s $BPE_MERGE_OPS > "${OUTPUT_DIR}/bpe.${BPE_MERGE_OPS}"

echo "Apply BPE with merge_ops=${BPE_MERGE_OPS}..."
for f in train dev test; do
  ${DATA_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${BPE_MERGE_OPS}" \
    < ${OUTPUT_DIR}/${f}.sources > ${OUTPUT_DIR}/${f}.bpe.${BPE_MERGE_OPS}.sources
  ${DATA_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${BPE_MERGE_OPS}" \
    < ${OUTPUT_DIR}/${f}.targets > ${OUTPUT_DIR}/${f}.bpe.${BPE_MERGE_OPS}.targets
done

# Learn vocabulary
cat "${OUTPUT_DIR}/train.bpe.${BPE_MERGE_OPS}.sources" "${OUTPUT_DIR}/train.bpe.${BPE_MERGE_OPS}.targets" \
  | ${DATA_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${BPE_MERGE_OPS}"
