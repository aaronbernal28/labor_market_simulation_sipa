"""
Script: run_simulation.py
Called by: workflow/rules/simulation.smk → rule run_simulation

Loads the aggregated transition series and runs the ABM simulation.
Full implementation TBD — currently produces a placeholder output.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

try:
	_sm = snakemake
except NameError:
	_sm = None

log_path = _sm.log[0] if _sm else "/dev/stderr"
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.labor_market import LaborMarketSimulation


def main():
	matrices_path = _sm.input.matrices if _sm else sys.argv[1]
	out_path = _sm.output.results if _sm else sys.argv[2]

	log.info("Loading transition series from %s", matrices_path)
	# TODO: deserialise multi-batch transition series from Parquet
	# transition_series = load_transition_series(matrices_path)

	log.info("Initialising LaborMarketSimulation (scaffold)")
	sim = LaborMarketSimulation(
		transition_series={},  # TODO: pass real data
		rate_fired=_sm.params.rate_fired if _sm else 0.02,
		rate_vacancies=_sm.params.rate_vacancies if _sm else 0.03,
		initial_employed_fraction=(
			_sm.params.initial_employed_fraction if _sm else 0.85
		),
		n_steps=_sm.params.n_steps if _sm else 120,
		seed=_sm.params.seed if _sm else 42,
	)

	log.info("Running simulation for %d steps", sim.n_steps)
	for _ in range(sim.n_steps):
		sim.step()
		if not sim.running:
			break

	# Placeholder output
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame({"step": [sim.current_step], "status": ["scaffold"]}).to_parquet(
		out_path, index=False
	)
	log.info("Saved simulation output to %s", out_path)


if __name__ == "__main__":
	main()
