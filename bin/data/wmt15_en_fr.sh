#! /usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_BASE="${OUTPUT_DIR}/wmt15_fr_en"
OUTPUT_DIR_DATA="${OUTPUT_DIR}/wmt15_fr_en/data"

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz \
  http://www.statmt.org/europarl/v7/fr-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

echo "Downloading UN corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/multiUN.fr.tgz \
  http://www.euromatrixplus.net/media/un-release/multiUN.fr.tgz
wget -nc -nv -O ${OUTPUT_DIR_DATA}/multiUN.en.tgz \
  http://www.euromatrixplus.net/media/un-release/multiUN.en.tgz

echo "Downloading News Commentary v10. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/nc-v10.tgz \
  http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz

echo "Downloading Gigaworld FR-EN. This may take a while..."
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/giga-fren.tar \
  http://www.statmt.org/wmt10/training-giga-fren.tar

echo "Downloading dev/test sets"
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/dev.tgz \
  http://www.statmt.org/wmt15/dev-v2.tgz
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz \
  http://www.statmt.org/wmt15/test.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"

mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"

mkdir -p "${OUTPUT_DIR_DATA}/multiUN"
mkdir -p "${OUTPUT_DIR_DATA}/multiUN/en"
mkdir -p "${OUTPUT_DIR_DATA}/multiUN/fr"
tar -xvzf "{OUTPUT_DIR_DATA}/multiUN.fr.tgz" -C "${OUTPUT_DIR_DATA}/multiUN/fr"
tar -xvzf "{OUTPUT_DIR_DATA}/multiUN.en.tgz" -C "${OUTPUT_DIR_DATA}/multiUN/en"

mkdir -p "${OUTPUT_DIR_DATA}/nc-v10"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v10.tgz" -C "${OUTPUT_DIR_DATA}/nc-v10"

mkdir -p "${OUTPUT_DIR_DATA}/giga-fren"
tar -xvzf "${OUTPUT_DIR_DATA}/giga-fren.tgz" -C "${OUTPUT_DIR_DATA}/giga-fren"

mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi