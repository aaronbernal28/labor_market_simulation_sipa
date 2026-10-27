import unittest

import numpy as np
import pandas as pd

from src.pipeline.bipartite_builder import BipartiteBuilder
from src.pipeline.transition_builder import BipartiteToTransition


class TestTransitionBuilder(unittest.TestCase):
	def setUp(self):
		# Create a concrete 3-sector + unemployed example matching docs/definitions.md
		# Workers 1..5 with uniform expansion weight pondera = 33.3333
		self.w = 33.3333
		self.universe = pd.DataFrame(
			{
				"id_trabajador": [1, 2, 3, 4, 5, 6],
				"pondera": [self.w] * 6,
			}
		)
		# At time t:
		# Workers 1, 2, 3 in sector '10' (N_t('10') = 3 * w)
		# Worker 4 in sector '20' (N_t('20') = 1 * w)
		# Workers 5, 6 not in states_t -> 'unemployed' (N_t('unemployed') = 2 * w)
		self.states_t = pd.DataFrame(
			{
				"id_trabajador": [1, 2, 3, 4],
				"sector": ["10", "10", "10", "20"],
				"pondera": [self.w] * 4,
			}
		)

		# At time t+1:
		# Worker 1 stays in '10'
		# Worker 2 moves '10' -> '20'
		# Worker 3 loses job '10' -> 'unemployed'
		# Worker 4 stays in '20'
		# Worker 5 gets hired 'unemployed' -> '20'
		# Worker 6 remains 'unemployed' -> 'unemployed'
		# Totals at t+1:
		#   N_{t+1}('10') = 1 * w (worker 1)
		#   N_{t+1}('20') = 3 * w (workers 2, 4, 5)
		#   N_{t+1}('unemployed') = 2 * w (workers 3, 6)
		self.states_t1 = pd.DataFrame(
			{
				"id_trabajador": [1, 2, 4, 5],
				"sector": ["10", "20", "20", "20"],
				"pondera": [self.w] * 4,
			}
		)

	def test_weighted_flows_and_conservation(self):
		builder = BipartiteToTransition(
			states_t=self.states_t,
			states_t1=self.states_t1,
			active_universe=self.universe,
			all_sectors=["10", "20"],
		)
		C = builder.compute_weighted_flows()

		self.assertEqual(list(C.index), ["unemployed", "10", "20"])
		self.assertEqual(list(C.columns), ["unemployed", "10", "20"])

		# Total weighted sum equals 6 * w
		self.assertAlmostEqual(C.values.sum(), 6 * self.w, places=4)

		# Specific cells
		self.assertAlmostEqual(C.loc["10", "10"], 1 * self.w, places=4)
		self.assertAlmostEqual(C.loc["10", "20"], 1 * self.w, places=4)
		self.assertAlmostEqual(C.loc["10", "unemployed"], 1 * self.w, places=4)
		self.assertAlmostEqual(C.loc["20", "20"], 1 * self.w, places=4)
		self.assertAlmostEqual(C.loc["unemployed", "20"], 1 * self.w, places=4)
		self.assertAlmostEqual(C.loc["unemployed", "unemployed"], 1 * self.w, places=4)

	def test_hidalgo_proximity_equation(self):
		builder = BipartiteToTransition(
			states_t=self.states_t,
			states_t1=self.states_t1,
			active_universe=self.universe,
			all_sectors=["10", "20"],
		)
		Phi = builder.compute_hidalgo_proximity()

		# Bounds check: 0 <= phi <= 1
		self.assertTrue((Phi.values >= 0.0).all())
		self.assertTrue((Phi.values <= 1.0).all())

		# Check phi_{t, '10', '20'}:
		# C('10', '20') = 1 * w
		# N_t('10') = 3 * w, N_{t+1}('20') = 3 * w -> max = 3 * w
		# phi = 1 / 3
		self.assertAlmostEqual(Phi.loc["10", "20"], 1.0 / 3.0, places=6)

		# Check phi_{t, '20', '20'}:
		# C('20', '20') = 1 * w
		# N_t('20') = 1 * w, N_{t+1}('20') = 3 * w -> max = 3 * w
		# phi = 1 / 3
		self.assertAlmostEqual(Phi.loc["20", "20"], 1.0 / 3.0, places=6)

		# Check phi_{t, 'unemployed', 'unemployed'}:
		# C('unemployed', 'unemployed') = 1 * w
		# N_t('unemployed') = 2 * w, N_{t+1}('unemployed') = 2 * w -> max = 2 * w
		# phi = 1 / 2
		self.assertAlmostEqual(Phi.loc["unemployed", "unemployed"], 0.5, places=6)

	def test_row_stochastic_probabilities(self):
		builder = BipartiteToTransition(
			states_t=self.states_t,
			states_t1=self.states_t1,
			active_universe=self.universe,
			all_sectors=["10", "20"],
		)
		T = builder.compute_row_stochastic()

		# Every row has active workers, so every row sums to 1.0
		np.testing.assert_allclose(T.sum(axis=1).values, [1.0, 1.0, 1.0])
		self.assertAlmostEqual(T.loc["10", "20"], 1.0 / 3.0, places=6)
		self.assertAlmostEqual(T.loc["20", "20"], 1.0, places=6)

	def test_bipartite_builder_age_and_max_rem_tot(self):
		# Observation period: 200012 (year = 2000)
		# Worker 10: two concurrent jobs in 200012 -> sectors '15' (rem 500) and '25' (rem 1200)
		#   Age = 2000 - 1960 = 40 <= 65 -> keeps '25'
		# Worker 11: born 1930 -> Age = 2000 - 1930 = 70 > 65 -> excluded!
		raw_df = pd.DataFrame(
			{
				"id_trabajador": [10, 10, 11],
				"rem_tot": [500.0, 1200.0, 900.0],
				"fnacim": [1960, 1960, 1930],
				"pondera": [33.33, 33.33, 33.33],
				"r32": ["15", "25", "15"],
			}
		)
		bb = BipartiteBuilder(
			df=raw_df,
			granularity=2,
			target_period=200012,
			max_working_age=65,
		)
		res = bb.extract_worker_states(raw_df["r32"])
		self.assertEqual(len(res), 1)
		self.assertEqual(res.iloc[0]["id_trabajador"], 10)
		self.assertEqual(res.iloc[0]["sector"], "25")


if __name__ == "__main__":
	unittest.main()
