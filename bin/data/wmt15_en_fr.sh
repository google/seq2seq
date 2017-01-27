#! /usr/bin/env bash

set -e

BASE_DIR=${BASE_DIR:-$HOME/nmt_data}
echo "Writing to ${BASE_DIR}. To change this, set the BASE_DIR environment variable."

OUTPUT_DIR="${OUTPUT_DIR}/wmt15_fr_en"
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
tar -xzf "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"

mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"

mkdir -p "${OUTPUT_DIR_DATA}/multiUN"
tar -xzf "${OUTPUT_DIR_DATA}/multiUN.fr.tgz" -C "${OUTPUT_DIR_DATA}/multiUN"
tar -xzf "${OUTPUT_DIR_DATA}/multiUN.en.tgz" -C "${OUTPUT_DIR_DATA}/multiUN"

mkdir -p "${OUTPUT_DIR_DATA}/nc-v10"
tar -xzf "${OUTPUT_DIR_DATA}/nc-v10.tgz" -C "${OUTPUT_DIR_DATA}/nc-v10"

mkdir -p "${OUTPUT_DIR_DATA}/giga-fren"
tar -xf "${OUTPUT_DIR_DATA}/giga-fren.tar" -C "${OUTPUT_DIR_DATA}/giga-fren"
gunzip "${OUTPUT_DIR_DATA}/giga-fren/giga-fren.release2.fixed.en.gz"
gunzip "${OUTPUT_DIR_DATA}/giga-fren/giga-fren.release2.fixed.fr.gz"

mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Clone Moses
# ========================================
if [ ! -d "${BASE_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${BASE_DIR}/mosesdecoder"
fi


# Prepare UN corpus
# ========================================
# Copy and rename files to use with Moses
cd ${OUTPUT_DIR_DATA}/multiUN/un && python ./extract.py en fr
mkdir -p "${OUTPUT_DIR_DATA}/multiUN/un/text/clean"
for f in $(find "${OUTPUT_DIR_DATA}/multiUN/un/text/en-fr" -name "*_en.snt"); do
  file_base=${f%_en.*}
  prefix=$(basename ${file_base})
  cp "${file_base}_en.snt" "${OUTPUT_DIR_DATA}/multiUN/un/text/clean/${prefix}.en"
  cp "${file_base}_fr.snt" "${OUTPUT_DIR_DATA}/multiUN/un/text/clean/${prefix}.fr"
done

# Clean UN corpus uses Moses
for f in ${OUTPUT_DIR_DATA}/multiUN/un/text/clean/*.en; do
  fbase=${f%.*}
  ${BASE_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase fr en "${fbase}.clean" 1 80
done

# Concatenate all cleaned data
find ${OUTPUT_DIR_DATA}/multiUN/un/text/clean -name "*clean.fr" | xargs cat > ${OUTPUT_DIR_DATA}/multiUN/un/text/all.clean.fr
find ${OUTPUT_DIR_DATA}/multiUN/un/text/clean -name "*clean.en" | xargs cat > ${OUTPUT_DIR_DATA}/multiUN/un/text/all.clean.en

# Prepare dev/test data
# ========================================

# Convert SGM files
for f in ${OUTPUT_DIR_DATA}/dev/dev/*.sgm; do
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl < $f > ${f%.*}
done
for f in ${OUTPUT_DIR_DATA}/test/test/*.sgm; do
  ${BASE_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl < $f > ${f%.*}
done

# Copy dev/test sets
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest{2009,2010,2011,2012,2013}.en ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest{2009,2010,2011,2012,2013}.fr ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-fren-ref.en ${OUTPUT_DIR}/newstest2014.en
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-fren-ref.fr ${OUTPUT_DIR}/newstest2014.fr
cp ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-enfr-src.en ${OUTPUT_DIR}/newstest2015.en
cp ${OUTPUT_DIR_DATA}/test/test/newsdiscusstest2015-enfr-ref.fr ${OUTPUT_DIR}/newstest2015.fr

# Concat all training data
# ========================================

for lang in fr en; do
  cat \
    "${OUTPUT_DIR_DATA}/europarl-v7-fr-en/europarl-v7.fr-en.${lang}" \
    "${OUTPUT_DIR_DATA}/common-crawl//commoncrawl.fr-en.${lang}" \
    "${OUTPUT_DIR_DATA}/nc-v10//news-commentary-v10.fr-en.${lang}" \
    "${OUTPUT_DIR_DATA}/multiUN/un/text/all.clean.${lang}" \
    "${OUTPUT_DIR_DATA}/giga-fren/giga-fren.release2.fixed.${lang}" \
    > "${OUTPUT_DIR}/train.${lang}"
done

# Tokenization and cleaning
# ========================================

# Tokenize data
for lang in en fr; do
  for f in ${OUTPUT_DIR}/*.${lang}; do
    echo "Tokenizing $f..."
    ${BASE_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l ${lang} -threads 8 < $f > ${f%.*}.tok.${lang}
  done
done

# Clean all tokenized corpora
for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${BASE_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase en fr "${fbase}.clean" 1 80
done


# Generate BPE data
# ========================================

# Clone Subword NMT
if [ ! -d "${BASE_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${BASE_DIR}/subword-nmt"
fi

for merge_ops in 32000; do

  # Learn BPE
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat \
    "${OUTPUT_DIR}/train.tok.clean.en" \
    "${OUTPUT_DIR}/train.tok.clean.fr" \
    | ${BASE_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  # Apply BPE
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en fr; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BASE_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  cat \
    "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" \
    "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.fr" \
    | ${BASE_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

