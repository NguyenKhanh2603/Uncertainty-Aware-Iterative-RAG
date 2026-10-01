"""Quantile interpolation and strict/inclusive boundary regression checks."""
import unittest
import numpy as np
from cce import cce_embedding_threshold, select as cce_select
from conflare import conflare_threshold, select as conflare_select
from traq import traq_retrieval_threshold, select as traq_select

class RulesTest(unittest.TestCase):
    def test_calibration_units_and_order_statistics(self):
        # Unequal number of supports distinguishes pooled pairs from per-query maxima.
        positive = [np.array([.1,.4,.9]),np.array([.6]),np.array([.8])]
        cutoff,audit = cce_embedding_threshold(positive,.1)
        self.assertAlmostEqual(cutoff,.1)
        self.assertEqual(audit['positive_pairs'],5)
        cutoff,audit = conflare_threshold(positive,.1)
        self.assertAlmostEqual(cutoff,.64)
        self.assertEqual(audit['calibration_records'],3)
        cutoff,audit = traq_retrieval_threshold(positive,.1)
        self.assertAlmostEqual(cutoff,.6)
        self.assertEqual(audit['alpha_retrieval'],.05)
    def test_exact_cutoff_ties(self):
        for selector,expected in [(cce_select,[False,True,True]),(conflare_select,[False,False,True]),(traq_select,[False,True,True])]:
            self.assertEqual(selector([.4,.5,.6],.5).tolist(),expected)
    def test_no_support_fails_calibration(self):
        for fit in [cce_embedding_threshold,conflare_threshold,traq_retrieval_threshold]:
            with self.assertRaises(ValueError): fit([np.array([])],.1)

if __name__=='__main__': unittest.main()
