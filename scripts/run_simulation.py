"""
Script: run_simulation.py
Called by: workflow/rules/simulation.smk → rule run_simulation

Loads the aggregated transition series and runs the ABM simulation.
Full implementation TBD — currently produces a placeholder output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.simulation.labor_market import LaborMarketSimulation

if TYPE_CHECKING:
	from snakemake.script import Snakemake

	snakemake: Snakemake

log_path: str = (
	snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else "/dev/stderr"
)
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log: logging.Logger = logging.getLogger(__name__)


def main() -> None:
	matrices_path: str = str(snakemake.input.matrices)
	out_path: str = str(snakemake.output.results)

	rate_fired: float = float(snakemake.params.rate_fired)
	rate_vacancies: float = float(snakemake.params.rate_vacancies)
	initial_employed_fraction: float = float(snakemake.params.initial_employed_fraction)
	n_steps: int = int(snakemake.params.n_steps)
	seed: int = int(snakemake.params.seed)

	log.info("Loading transition series from %s", matrices_path)

	log.info("Initialising LaborMarketSimulation (scaffold)")
	sim: LaborMarketSimulation = LaborMarketSimulation(
		transition_series={},  # TODO: pass real data
		rate_fired=rate_fired,
		rate_vacancies=rate_vacancies,
		initial_employed_fraction=initial_employed_fraction,
		n_steps=n_steps,
		seed=seed,
	)

	log.info("Running simulation for %d steps", sim.n_steps)
	for _ in range(sim.n_steps):
		sim.step()
		if not sim.running:
			break

	# Placeholder output
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	placeholder_df: pd.DataFrame = pd.DataFrame(
		{
			"step": list(range(sim.current_step)),
			"status": ["placeholder"] * sim.current_step,
		}
	)
	placeholder_df.to_parquet(out_path, index=False)
	log.info("Saved placeholder simulation output to %s", out_path)


if __name__ == "__main__":
	main()
