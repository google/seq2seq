#! /usr/bin/env python

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
Processes a CNN/Daily Mail story file into a format that can
be used for summarization.
"""

import fileinput
import re

def process_story(text):
  """Processed a story text into an (article, summary) tuple.
  """
  # Split by highlights
  elements = text.split("@highlight")
  elements = [_.strip() for _ in elements]

  story_text = elements[0]
  highlights = elements[1:]

  # Join all highlights into a single blob
  highlights_joined = "; ".join(highlights)
  highlights_joined = re.sub(r"\s+", " ", highlights_joined)
  highlights_joined = highlights_joined.strip()

  # Remove newlines from story
  # story_text = story_text.replace("\n", " ")
  story_text = re.sub(r"\s+", " ", story_text)
  story_text = story_text.strip()

  return story_text, highlights_joined

def main(*args, **kwargs):
  """Program entry point"""
  story_text = "\n".join(list(fileinput.input()))
  story, highlights = process_story(story_text)

  if story and highlights:
    print("{}\t{}".format(story, highlights))

if __name__ == '__main__':
  main()
