# -*- coding: utf-8 -*-

"""Tests for Metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from seq2seq.metrics import bleu
from seq2seq.metrics import rouge
from seq2seq.metrics.metric_specs import BleuMetricSpec


class TestMosesBleu(tf.test.TestCase):
  """Tests using the Moses multi-bleu script to calcualte BLEU score"""

  def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
    """Runs a multi-bleu test."""
    result = bleu.moses_multi_bleu(
        hypotheses=hypotheses,
        references=references,
        lowercase=lowercase)
    np.testing.assert_almost_equal(result, expected_bleu, decimal=2)


  def test_multi_bleu(self):
    self._test_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over the dog 笑",
            "The brown fox jumps over the dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"]),
        lowercase=False,
        expected_bleu=46.51)

  def test_multi_bleu_lowercase(self):
    self._test_multi_bleu(
        hypotheses=np.array([
            "The brown fox jumps over The Dog 笑",
            "The brown fox jumps over The Dog 2 笑"]),
        references=np.array([
            "The quick brown fox jumps over the lazy dog 笑",
            "The quick brown fox jumps over the lazy dog 笑"]),
        lowercase=True,
        expected_bleu=46.51)


class TestBleuMetricSpec(tf.test.TestCase):
  """Tests the `PrintModelAnalysisHook` hook"""

  def test_bleu(self):
    predictions = {
        "predicted_tokens": tf.placeholder(dtype=tf.string)
    }
    labels = {
        "target_tokens": tf.placeholder(dtype=tf.string)
    }

    metric_spec = BleuMetricSpec()
    value, update_op = metric_spec.create_metric_ops(None, labels, predictions)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      hypotheses = ["A B C D E F", "A B C D E F"]
      references = ["A B C D E F", "A B A D E F"]

      scores = []
      for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split(" ")
        ref = ref.split(" ")
        sess.run(update_op, {
            predictions["predicted_tokens"]: [hyp],
            labels["target_tokens"]: [ref]
        })
        scores.append(sess.run(value))

      np.testing.assert_almost_equal(scores[0], 100.0, decimal=2)
      np.testing.assert_almost_equal(scores[1], 69.19, decimal=2)


class TestRougeMetric(tf.test.TestCase):
  def test_get_ngrams(self):
    self.assertTrue(not rouge._get_ngrams(3, ""))
    correct_ngrams = [("t", "e"), ("e", "s"), ("s", "t"),
                      ("t", "i"), ("i", "n"), ("n", "g")]
    found_ngrams = rouge._get_ngrams(2, "testing")
    self.assertEqual(len(correct_ngrams), len(found_ngrams))
    for ngram in correct_ngrams:
      self.assertTrue(ngram in found_ngrams)

  def test_split_into_words(self):
    sentences1 = ["One two two Two Three"]
    self.assertEqual(
        ["One", "two", "two", "Two", "Three"],
        rouge._split_into_words(sentences1))

  def test_get_word_ngrams(self):
    sentences = ["This is a test"]
    correct_ngrams = [("This", "is"), ("is", "a"), ("a", "test")]
    found_ngrams = rouge._get_word_ngrams(2, sentences)
    for ngram in correct_ngrams:
      self.assertTrue(ngram in found_ngrams)

  def test_len_lcs(self):
      self.assertEqual(rouge._len_lcs("1234", "1224533324"), 4)
      self.assertEqual(rouge._len_lcs("thisisatest", "testing123testing"), 7)

  def test_recon_lcs(self):
    self.assertEqual(
        rouge._recon_lcs("1234", "1224533324"),
        ("1", "2", "3", "4"))
    self.assertEqual(
        rouge._recon_lcs("thisisatest", "testing123testing"),
        ("t", "s", "i", "t", "e", "s", "t"))

  def test_rouge_n(self):
    candidate = ["pulses may ease schizophrenic voices"]

    reference1 = ["magnetic pulse series sent through brain may " \
      "ease schizophrenic voices"]

    reference2 = ["yale finds magnetic stimulation some relief to " \
      "schizophrenics imaginary voices"]

    self.assertAlmostEqual(rouge.rouge_n(candidate, reference1, 1), 4/10)
    self.assertAlmostEqual(rouge.rouge_n(candidate, reference2, 1), 1/10)

    self.assertAlmostEqual(rouge.rouge_n(candidate, reference1, 2), 3/9)
    self.assertAlmostEqual(rouge.rouge_n(candidate, reference2, 2), 0/9)

    self.assertAlmostEqual(rouge.rouge_n(candidate, reference1, 3), 2/8)
    self.assertAlmostEqual(rouge.rouge_n(candidate, reference2, 3), 0/8)

    self.assertAlmostEqual(rouge.rouge_n(candidate, reference1, 4), 1/7)
    self.assertAlmostEqual(rouge.rouge_n(candidate, reference2, 4), 0/7)

    # These tests will apply when multiple reference summaries can be input
    # self.assertAlmostEqual(
    #     rouge.rouge_n(candidate, [reference1, reference2], 1), 5/20)
    # self.assertAlmostEqual(
    #     rouge.rouge_n(candidate, [reference1, reference2], 2), 3/18)
    # self.assertAlmostEqual(
    #     rouge.rouge_n(candidate, [reference1, reference2], 3), 2/16)
    # self.assertAlmostEqual(
    #     rouge.rouge_n(candidate, [reference1, reference2], 4), 1/14)


  def test_rouge_l_sentence_level(self):
    reference = ["police killed the gunman"]
    candidate1 = ["police kill the gunman"]
    candidate2 = ["the gunman kill police"]
    candidate3 = ["the gunman police killed"]

    self.assertAlmostEqual(
        rouge.rouge_l_sentence_level(candidate1, reference), 3/4)
    self.assertAlmostEqual(
        rouge.rouge_l_sentence_level(candidate2, reference), 2/4)
    self.assertAlmostEqual(
        rouge.rouge_l_sentence_level(candidate3, reference), 2/4)

  def test_union_lcs(self):
    reference = ["one two three four five"]
    candidates = ["one two six seven eight", "one three eight nine five"]
    self.assertAlmostEqual(rouge._union_lcs(candidates, reference[0]), 4/5)

  def test_rouge_l_summary_level(self):
    reference = ["one two three four five", "one two three four five"]
    candidates = ["one two six seven eight", "one three eight nine five"]
    self.assertAlmostEqual(
        rouge.rouge_l_summary_level(candidates, reference), 0.16)


if __name__ == "__main__":
  tf.test.main()
