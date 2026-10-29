"""
Script: compute_empirical_series.py
Called by: workflow/rules/simulation.smk -> rule compute_empirical_series

Computes the empirical macroeconomic state vector S_t, Shannon entropy H_t,
net sectoral variations Delta x_t, and Lilien reallocation index L_t
across the empirical SIPA panel (Track 1).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.empirical_series import (
	build_empirical_series,
	compute_period_headcounts,
)

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
	states_paths: list[str] = [str(p) for p in snakemake.input.states]
	registry_path: str = str(snakemake.input.registry)
	macro_out_path: str = str(snakemake.output.macro)
	sectoral_out_path: str = str(snakemake.output.sectoral)

	periods: list[int] = [int(p) for p in snakemake.params.periods]
	max_working_age: int = int(snakemake.params.max_working_age)

	log.info("Loading cumulative worker registry from %s", registry_path)
	registry: pd.DataFrame = pd.read_parquet(registry_path)

	headcounts_by_period: dict[int, pd.Series] = {}

	# Align states files with period keys
	# States paths correspond to expand(".../{period}/bipartite_g{gran}.parquet", period=PERIODS)
	period_to_path: dict[int, str] = dict(zip(periods, states_paths, strict=True))

	for period in periods:
		path: str = period_to_path[period]
		log.info("Processing period %d from %s", period, path)
		states_t: pd.DataFrame = pd.read_parquet(path)

		obs_year: int = period // 100
		age: pd.Series = obs_year - registry["fnacim"]
		active_mask: pd.Series = (registry["first_period"] <= period) & (
			age <= max_working_age
		)
		active_universe: pd.DataFrame = registry.loc[
			active_mask, ["id_trabajador", "pondera"]
		]

		counts: pd.Series = compute_period_headcounts(
			states=states_t,
			active_universe=active_universe,
			unemployed_label="unemployed",
		)
		headcounts_by_period[period] = counts

	log.info(
		"Building empirical series tables across %d periods",
		len(headcounts_by_period),
	)
	macro_df, sectoral_df = build_empirical_series(
		headcounts_by_period=headcounts_by_period,
		unemployed_label="unemployed",
	)

	Path(macro_out_path).parent.mkdir(parents=True, exist_ok=True)
	Path(sectoral_out_path).parent.mkdir(parents=True, exist_ok=True)

	macro_df.to_csv(macro_out_path, index=False)
	sectoral_df.to_csv(sectoral_out_path, index=False)

	log.info("Saved empirical macro series to %s", macro_out_path)
	log.info("Saved empirical sectoral series to %s", sectoral_out_path)


if __name__ == "__main__":
	main()
