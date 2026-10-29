import unittest

import numpy as np
import pandas as pd

from src.pipeline.empirical_series import (
	build_empirical_series,
	compute_lilien_index,
	compute_period_headcounts,
	compute_shannon_entropy,
	compute_shares,
)


class TestEmpiricalSeries(unittest.TestCase):
	def test_shares_and_entropy(self):
		counts = pd.Series({"unemployed": 200.0, "A": 400.0, "B": 400.0})
		shares = compute_shares(counts)
		self.assertAlmostEqual(shares.sum(), 1.0)
		self.assertAlmostEqual(shares["unemployed"], 0.2)
		self.assertAlmostEqual(shares["A"], 0.4)

		# Entropy of uniform 2 states
		uniform_2 = pd.Series({"A": 0.5, "B": 0.5})
		self.assertAlmostEqual(compute_shannon_entropy(uniform_2), 1.0)

		# Degenerate state
		degenerate = pd.Series({"A": 1.0, "B": 0.0})
		self.assertAlmostEqual(compute_shannon_entropy(degenerate), 0.0)

	def test_lilien_index_proportional_growth(self):
		# When all formal sectors grow at the exact same rate, Lilien index must be 0.0
		h0 = pd.Series({"unemployed": 100.0, "A": 200.0, "B": 300.0})
		h1 = pd.Series({"unemployed": 120.0, "A": 400.0, "B": 600.0})  # 2x growth
		l_val = compute_lilien_index(h0, h1)
		self.assertAlmostEqual(l_val, 0.0, places=7)

	def test_lilien_index_asymmetric_reallocation(self):
		# Sector A doubles, Sector B halves
		h0 = pd.Series({"unemployed": 100.0, "A": 100.0, "B": 100.0})
		h1 = pd.Series({"unemployed": 100.0, "A": 200.0, "B": 50.0})
		l_val = compute_lilien_index(h0, h1)
		self.assertGreater(l_val, 0.0)

	def test_period_headcounts_with_unemployed(self):
		universe = pd.DataFrame(
			{
				"id_trabajador": [1, 2, 3, 4],
				"pondera": [10.0, 10.0, 10.0, 10.0],
			}
		)
		# Workers 1 and 2 are employed; 3 and 4 are unemployed
		states = pd.DataFrame(
			{
				"id_trabajador": [1, 2],
				"sector": ["A", "B"],
				"pondera": [10.0, 10.0],
			}
		)
		counts = compute_period_headcounts(states, universe)
		self.assertEqual(counts["unemployed"], 20.0)
		self.assertEqual(counts["A"], 10.0)
		self.assertEqual(counts["B"], 10.0)
		self.assertEqual(counts.sum(), 40.0)

	def test_build_empirical_series(self):
		h0 = pd.Series({"unemployed": 50.0, "A": 50.0})
		h1 = pd.Series({"unemployed": 40.0, "A": 60.0})
		headcounts = {199601: h0, 199701: h1}

		macro_df, sectoral_df = build_empirical_series(headcounts)
		self.assertEqual(len(macro_df), 2)
		self.assertTrue(np.isnan(macro_df.loc[0, "lilien_index"]))
		self.assertFalse(np.isnan(macro_df.loc[1, "lilien_index"]))
		self.assertEqual(macro_df.loc[0, "unemployment_rate"], 0.5)
		self.assertEqual(macro_df.loc[1, "unemployment_rate"], 0.4)

		self.assertEqual(len(sectoral_df), 4)  # 2 periods x 2 sectors


if __name__ == "__main__":
	unittest.main()
