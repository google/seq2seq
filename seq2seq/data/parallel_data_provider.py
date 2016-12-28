"""A Data Provder that reads parallel (aligned) data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner


class ParallelDataProvider(data_provider.DataProvider):
  """Creates a ParallelDataProvider. This data provider reads two datasets
  in parallel, keeping them aligned.

  Args:
    dataset1: The first dataset. An instance of the Dataset class.
    dataset2: The second dataset. An instance of the Dataset class.
      Can be None. If None, only `dataset1` is read.
    num_readers: The number of parallel readers to use.
    shuffle: Whether to shuffle the data sources and common queue when
      reading.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.
    common_queue_capacity: The capacity of the common queue.
    common_queue_min: The minimum number of elements in the common queue after
      a dequeue.
    seed: The seed to use if shuffling.
  """

  def __init__(self,
               dataset1,
               dataset2,
               shuffle=True,
               num_epochs=None,
               common_queue_capacity=4096,
               common_queue_min=1024,
               seed=None):

    if seed is None:
      seed = np.random.randint(10e8)

    _, data_source = parallel_reader.parallel_read(
        dataset1.data_sources,
        reader_class=dataset1.reader,
        num_epochs=num_epochs,
        num_readers=1,
        shuffle=False,
        capacity=common_queue_capacity,
        min_after_dequeue=common_queue_min,
        seed=seed)

    data_target = ""
    if dataset2 is not None:
      _, data_target = parallel_reader.parallel_read(
          dataset2.data_sources,
          reader_class=dataset2.reader,
          num_epochs=num_epochs,
          num_readers=1,
          shuffle=False,
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          seed=seed)

    # Optionally shuffle the data
    if shuffle:
      shuffle_queue = data_flow_ops.RandomShuffleQueue(
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          dtypes=[tf.string, tf.string],
          seed=seed)
      enqueue_ops = []
      enqueue_ops.append(shuffle_queue.enqueue([data_source, data_target]))
      queue_runner.add_queue_runner(
          queue_runner.QueueRunner(shuffle_queue, enqueue_ops))
      data_source, data_target = shuffle_queue.dequeue()

    # Decode source items
    items = dataset1.decoder.list_items()
    tensors = dataset1.decoder.decode(data_source, items)

    if dataset2 is not None:
      # Decode target items
      items2 = dataset2.decoder.list_items()
      tensors2 = dataset2.decoder.decode(data_target, items2)

      # Merge items and results
      items = items + items2
      tensors = tensors + tensors2

    super(ParallelDataProvider, self).__init__(
        items_to_tensors=dict(zip(items, tensors)),
        num_samples=dataset1.num_samples)
