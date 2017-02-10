# -*- coding: utf-8 -*-

"""BLEU metric implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import shutil
import numpy as np
from pyrouge import Rouge155

import tensorflow as tf

def rouge(hypotheses, references):
  """Calculate the ROUGE score for hypotheses and references.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.

  Returns:
    A dictionary corresponding to ROUGE scores. See
    https://pypi.python.org/pypi/pyrouge/ for details.
  """

  if np.size(hypotheses) == 0:
    return {}


  # Create a new temporary directory for rouge files
  hyp_dir = tempfile.mkdtemp(prefix="rouge_hyp")
  ref_dir = tempfile.mkdtemp(prefix="rouge_ref")

  # Dump hypotheses and references to files
  for i, hyp in enumerate(hypotheses):
    with open(os.path.join(hyp_dir, "{:06d}.txt".format(i)), "wb") as file:
      file.write(hyp.encode("utf-8"))
  for i, hyp in enumerate(references):
    with open(os.path.join(ref_dir, "{:06d}.txt".format(i)), "wb") as file:
      file.write(hyp.encode("utf-8"))

  # Calculate ROUGE
  try:
    rouge155 = Rouge155()
    rouge155.model_dir = hyp_dir
    rouge155.system_dir = ref_dir
    rouge155.model_filename_pattern = "#ID#.txt"
    rouge155.system_filename_pattern = r"(\d+).txt"
    output = rouge155.convert_and_evaluate()
    output_dict = rouge155.output_to_dict(output)
  except Exception as e:
    output_dict = {}
    tf.logging.warning("ROUGE script failed")
    tf.logging.warning(e)

  shutil.rmtree(hyp_dir, ignore_errors=True)
  shutil.rmtree(ref_dir, ignore_errors=True)

  return output_dict
