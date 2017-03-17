# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A collection of commonly used post-processing functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def strip_bpe(text):
  """Deodes text that was processed using BPE from
  https://github.com/rsennrich/subword-nmt"""
  return text.replace("@@ ", "").strip()

def decode_sentencepiece(text):
  """Decodes text that uses https://github.com/google/sentencepiece encoding.
  Assumes that pieces are separated by a space"""
  return "".join(text.split(" ")).replace("▁", " ").strip()

def slice_text(text,
               eos_token="SEQUENCE_END",
               sos_token="SEQUENCE_START"):
  """Slices text from SEQUENCE_START to SEQUENCE_END, not including
  these special tokens.
  """
  eos_index = text.find(eos_token)
  text = text[:eos_index] if eos_index > -1 else text
  sos_index = text.find(sos_token)
  text = text[sos_index+len(sos_token):] if sos_index > -1 else text
  return text.strip()
