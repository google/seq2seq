#! /usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data/gigaword_data}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."
mkdir -p $OUTPUT_DIR

# Unzip all files
find  $GIGAWORD_DIR -name *.gz | xargs gunzip

# Process all data files
for f in $(find $GIGAWORD_DIR -type f | grep data/); do
  outfile="$OUTPUT_DIR/$(basename $f)"
  $BASE_DIR/bin/data/gigaword_summarization.py -f $f -o $outfile
  echo $outfile
done

