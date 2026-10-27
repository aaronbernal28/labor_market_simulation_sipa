"""
Script: compute_transition.py
Called by: workflow/rules/transition.smk -> rule compute_transition

Loads two period worker state files (start and end) and the cumulative worker
registry, filters to the working-age cumulative labor force up to end,
and computes the (N+1) x (N+1) Hidalgo proximity / transition matrix.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.transition_builder import BipartiteToTransition

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
	current_path: str = str(snakemake.input.current)
	next_path: str = str(snakemake.input.next_b)
	registry_path: str = str(snakemake.input.registry)
	out_path: str = str(snakemake.output.matrix)

	t_start: int = int(snakemake.params.start)
	t_end: int = int(snakemake.params.end)
	metric: str = str(snakemake.params.metric)
	max_working_age: int = int(snakemake.params.max_working_age)

	obs_year_end: int = t_end // 100

	log.info(
		"Computing %s matrix for transition (%d -> %d)",
		metric,
		t_start,
		t_end,
	)

	states_t: pd.DataFrame = pd.read_parquet(current_path)
	states_t1: pd.DataFrame = pd.read_parquet(next_path)
	registry: pd.DataFrame = pd.read_parquet(registry_path)

	# Filter cumulative labor force:
	# 1. Entered formal market on or before t_end (first_period <= t_end)
	# 2. Working age at t_end (obs_year_end - fnacim <= max_working_age)
	age_end: pd.Series = obs_year_end - registry["fnacim"]
	active_mask: pd.Series = (registry["first_period"] <= t_end) & (
		age_end <= max_working_age
	)
	active_universe: pd.DataFrame = registry.loc[
		active_mask, ["id_trabajador", "pondera"]
	].copy()

	log.info(
		"Active working-age cumulative pool at %d: %d workers",
		t_end,
		len(active_universe),
	)

	builder: BipartiteToTransition = BipartiteToTransition(
		states_t=states_t,
		states_t1=states_t1,
		active_universe=active_universe,
	)
	matrix_df: pd.DataFrame = builder.compute(metric=metric)

	log.info(
		"Computed %s matrix of shape %d x %d",
		metric,
		matrix_df.shape[0],
		matrix_df.shape[1],
	)

	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	matrix_df.to_parquet(out_path, index=True)
	log.info("Saved matrix to %s", out_path)


if __name__ == "__main__":
	main()
