#! /usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data/gigaword}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."
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
find $OUTPUT_DIR/extracted -name sources.txt | sort | xargs cat > $OUTPUT_DIR/combined.sources.txt
find $OUTPUT_DIR/extracted -name targets.txt | sort | xargs cat > $OUTPUT_DIR/combined.targets.txt

echo "Found $(wc -l $OUTPUT_DIR/combined.sources.txt) sources."
echo "Found $(wc -l $OUTPUT_DIR/combined.targets.txt) targets."

# Tokenize using MOSES
# TODO

# Clean dataset
# TODO

# Learn BPE
# TODO
