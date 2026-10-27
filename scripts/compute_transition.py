"""
Script: compute_transition.py
Called by: workflow/rules/transition.smk -> rule compute_transition

Loads two consecutive batch worker state files and the cumulative worker
registry, filters to the working-age cumulative labor force up to t+1,
and computes the (N+1) x (N+1) Hidalgo proximity / transition matrix.
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
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.transition_builder import BipartiteToTransition


def _infer_end_period_from_path(bipartite_path: str | Path) -> int:
	"""Extract the end period YYYYMM from '.../{start}_{end}/bipartite_g{gran}.parquet'."""
	batch_folder = Path(bipartite_path).parent.name
	parts = batch_folder.split("_")
	return int(parts[1])


def main():
	current_path = _sm.input.current if _sm else sys.argv[1]
	next_path = _sm.input.next_b if _sm else sys.argv[2]
	registry_path = _sm.input.registry if _sm else sys.argv[3]
	out_path = _sm.output.matrix if _sm else sys.argv[4]

	metric = _sm.params.metric if _sm else "hidalgo"
	max_working_age = int(_sm.params.max_working_age) if _sm else 65

	t_end = _infer_end_period_from_path(current_path)
	t1_end = _infer_end_period_from_path(next_path)
	obs_year_t1 = t1_end // 100

	log.info(
		"Computing %s matrix for transition (%d -> %d)",
		metric,
		t_end,
		t1_end,
	)

	states_t = pd.read_parquet(current_path)
	states_t1 = pd.read_parquet(next_path)
	registry = pd.read_parquet(registry_path)

	# Filter cumulative labor force:
	# 1. Entered formal market on or before t1_end (first_period <= t1_end)
	# 2. Working age at t+1 (obs_year_t1 - fnacim <= max_working_age)
	age_t1 = obs_year_t1 - registry["fnacim"]
	active_mask = (registry["first_period"] <= t1_end) & (
		age_t1 <= max_working_age
	)
	active_universe = registry.loc[
		active_mask, ["id_trabajador", "pondera"]
	].copy()

	log.info(
		"Active working-age cumulative pool at %d: %d workers",
		t1_end,
		len(active_universe),
	)

	builder = BipartiteToTransition(
		states_t=states_t,
		states_t1=states_t1,
		active_universe=active_universe,
	)
	matrix_df = builder.compute(metric=metric)

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
