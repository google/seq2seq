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

# Tokenize using MOSES
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
# for f in ${OUTPUT_DIR}/*.de; do
#   echo "Tokenizing $f..."
#   ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
# done

# Clean dataset
# TODO

# Learn BPE
# TODO
