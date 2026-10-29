import random
import unittest

import pandas as pd

from src.pipeline.probability_normalizer import normalize_rows
from src.simulation.labor_market import LaborMarketModel
from src.simulation.perturbations import (
	apply_mass_layoff,
	apply_sector_contraction,
	build_baseline_sectors,
	build_uniform_sectors,
	resolve_target_sector,
)


class TestSimulation(unittest.TestCase):
	def test_normalize_rows(self):
		matrix = pd.DataFrame(
			[[2.0, 2.0], [0.0, 0.0]],
			index=["A", "B"],
			columns=["A", "B"],
		)
		probs = normalize_rows(matrix)
		self.assertAlmostEqual(probs.loc["A", "A"], 0.5)
		self.assertAlmostEqual(probs.loc["A", "B"], 0.5)
		# Zero-sum row receives 1.0 on diagonal (stays in sector)
		self.assertAlmostEqual(probs.loc["B", "B"], 1.0)
		self.assertAlmostEqual(probs.loc["B", "A"], 0.0)

	def test_perturbation_baseline_allocation(self):
		shares = pd.Series({"unemployed": 0.4, "A": 0.35, "B": 0.25})
		sectors = build_baseline_sectors(shares, n_agents=1000)
		self.assertEqual(len(sectors), 1000)
		self.assertEqual(sectors.count("unemployed"), 400)
		self.assertEqual(sectors.count("A"), 350)
		self.assertEqual(sectors.count("B"), 250)

	def test_perturbation_mass_layoff(self):
		# 100 unemployed, 900 formal
		initial = ["unemployed"] * 100 + ["A"] * 500 + ["B"] * 400
		rng = random.Random(42)
		post_layoff = apply_mass_layoff(initial, fraction=0.20, rng=rng)
		self.assertEqual(len(post_layoff), 1000)
		# 20% of 900 = 180 laid off -> 100 + 180 = 280 unemployed
		self.assertEqual(post_layoff.count("unemployed"), 280)

	def test_perturbation_sector_contraction(self):
		initial = ["unemployed"] * 200 + ["A"] * 400 + ["B"] * 400
		rng = random.Random(42)
		contracted = apply_sector_contraction(
			initial, target="A", fraction=0.50, rng=rng
		)
		self.assertEqual(len(contracted), 1000)
		# 50% of 400 = 200 contracted from A -> A now has 200
		self.assertEqual(contracted.count("A"), 200)
		# Displaced 200 moved to B
		self.assertEqual(contracted.count("B"), 600)

	def test_perturbation_uniform(self):
		sectors = build_uniform_sectors(["unemployed", "A", "B", "C"], n_agents=100)
		self.assertEqual(len(sectors), 100)
		for sec in ["unemployed", "A", "B", "C"]:
			self.assertEqual(sectors.count(sec), 25)

	def test_resolve_target_sector(self):
		shares = pd.Series({"unemployed": 0.5, "Retail": 0.3, "Agri": 0.2})
		self.assertEqual(resolve_target_sector(shares, "auto"), "Retail")
		self.assertEqual(resolve_target_sector(shares, "Agri"), "Agri")

	def test_model_deterministic_transition(self):
		# Deterministic cycle: unemployed -> A -> unemployed
		P0 = pd.DataFrame(
			[[0.0, 1.0], [0.0, 1.0]],
			index=["unemployed", "A"],
			columns=["unemployed", "A"],
		)
		P1 = pd.DataFrame(
			[[1.0, 0.0], [1.0, 0.0]],
			index=["unemployed", "A"],
			columns=["unemployed", "A"],
		)
		seq = [(199701, P0), (199801, P1)]
		initial = ["unemployed"] * 50

		model = LaborMarketModel(
			probs_sequence=seq,
			initial_sectors=initial,
			start_period=199601,
			rng=42,
		)
		self.assertEqual(model.current_unemployed, 50)

		# Step 1: all move to A
		model.step()
		self.assertEqual(model.current_unemployed, 0)
		self.assertEqual(model.current_formal_employed, 50)

		# Step 2: all move to unemployed
		model.step()
		self.assertEqual(model.current_unemployed, 50)
		self.assertEqual(model.current_formal_employed, 0)

		macro_df = model.datacollector.get_model_vars_dataframe()
		self.assertEqual(len(macro_df), 3)  # step 0, 1, 2


if __name__ == "__main__":
	unittest.main()
