"""
Script: compute_transition.py
Called by: workflow/rules/transition.smk -> rule compute_transition

Loads two consecutive batch worker state files and the cumulative worker
registry, filters to the working-age cumulative labor force up to t+1,
and computes the (N+1) x (N+1) Hidalgo proximity / transition matrix.
"""

from __future__ import annotations

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


def _infer_end_period_from_path(bipartite_path: str | Path) -> int:
	"""Extract the end period YYYYMM from '.../{start}_{end}/bipartite_g{gran}.parquet'."""
	batch_folder: str = Path(bipartite_path).parent.name
	parts: list[str] = batch_folder.split("_")
	return int(parts[1])


def main() -> None:
	current_path: str = str(snakemake.input.current)
	next_path: str = str(snakemake.input.next_b)
	registry_path: str = str(snakemake.input.registry)
	out_path: str = str(snakemake.output.matrix)

	metric: str = str(snakemake.params.metric)
	max_working_age: int = int(snakemake.params.max_working_age)

	t_end: int = _infer_end_period_from_path(current_path)
	t1_end: int = _infer_end_period_from_path(next_path)
	obs_year_t1: int = t1_end // 100

	log.info(
		"Computing %s matrix for transition (%d -> %d)",
		metric,
		t_end,
		t1_end,
	)

	states_t: pd.DataFrame = pd.read_parquet(current_path)
	states_t1: pd.DataFrame = pd.read_parquet(next_path)
	registry: pd.DataFrame = pd.read_parquet(registry_path)

	# Filter cumulative labor force:
	# 1. Entered formal market on or before t1_end (first_period <= t1_end)
	# 2. Working age at t+1 (obs_year_t1 - fnacim <= max_working_age)
	age_t1: pd.Series = obs_year_t1 - registry["fnacim"]
	active_mask: pd.Series = (registry["first_period"] <= t1_end) & (
		age_t1 <= max_working_age
	)
	active_universe: pd.DataFrame = registry.loc[
		active_mask, ["id_trabajador", "pondera"]
	].copy()

	log.info(
		"Active working-age cumulative pool at %d: %d workers",
		t1_end,
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
