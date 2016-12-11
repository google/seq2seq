#! /usr/bin/env python
"""
Generates a TFRecords file given sequence-aligned source and target files.

Example Usage:

python ./generate_examples.py --source_file <SOURCE_FILE> \
  --target_file <TARGET_FILE> \
  --output_file <OUTPUT_FILE>
"""

import tensorflow as tf
from tensorflow.python.platform import gfile

tf.flags.DEFINE_string('source_file', None,
                       'File containing content in source language.')
tf.flags.DEFINE_string(
    'target_file', None,
    'File containing content in target language, parallel line by line to the'
    'source file.')
tf.flags.DEFINE_string('output_file', None,
                       'File to output tf.Example TFRecords.')

FLAGS = tf.flags.FLAGS


def build_example(pair_id, source, target):
  """Transforms pair of 'source' and 'target' strings into a tf.Example.

  Assumes that 'source' and 'target' are already tokenized.

  Args:
    pair_id: id of this pair of source and target strings.
    source: a pretokenized source string.
    target: a pretokenized target string.

  Returns:
    a tf.Example corresponding to the 'source' and 'target' inputs.
  """
  pair_id = str(pair_id)
  source_tokens = source.strip().split(' ')
  target_tokens = target.strip().split(' ')
  ex = tf.train.Example()

  ex.features.feature['pair_id'].bytes_list.value.append(
      pair_id.encode('utf-8'))
  ex.features.feature['source_len'].int64_list.value.append(len(source_tokens))
  ex.features.feature['target_len'].int64_list.value.append(len(target_tokens))

  source_tokens = [t.encode('utf-8') for t in source_tokens]
  target_tokens = [t.encode('utf-8') for t in target_tokens]

  ex.features.feature['source_tokens'].bytes_list.value.extend(source_tokens)
  ex.features.feature['target_tokens'].bytes_list.value.extend(target_tokens)

  return ex


def write_tfrecords(examples, output_file):
  """Writes a list of tf.Examples to 'output_file'.

  Args:
    examples: An iterator of tf.Example records
    outputfile: path to the output file
  """
  writer = tf.python_io.TFRecordWriter(output_file)
  print('Creating TFRecords file at {}...'.format(output_file))
  for row in examples:
    writer.write(row.SerializeToString())
  writer.close()
  print('Wrote to {}'.format(output_file))


def generate_examples(source_file, target_file):
  """Creates an iterator of tf.Example records given aligned
  source and target files.

  Args:
    source_file: path to file with newline-separated source strings
    target_file: path to file with newline-separated target strings

  Returns:
    An iterator of tf.Example objects.
  """
  i = 0
  with gfile.GFile(source_file) as source_records:
    with gfile.GFile(target_file) as target_records:
      for i, (source, target) in enumerate(zip(source_records, target_records)):
        if i % 10000 == 0:
          print('Processed {} records'.format(i + 1))
        yield build_example(i, source, target)
      print('Processed {} records. Done.'.format(i + 1))


def main(unused_argv):
  """Main function.
  """
  #pylint: disable=unused-argument
  examples = generate_examples(FLAGS.source_file, FLAGS.target_file)
  write_tfrecords(examples, FLAGS.output_file)


if __name__ == '__main__':
  tf.app.run()
