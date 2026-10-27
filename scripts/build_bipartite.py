"""
Script: build_bipartite.py
Called by: workflow/rules/bipartite.smk -> rule build_bipartite

Loads the partitioned MLER Parquet slice at the batch end month (t_end),
applies working-age filtering and primary job selection (max rem_tot),
and saves the resolved worker states to Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.bipartite_builder import BipartiteBuilder
from src.pipeline.data_loader import MLERLoader

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
	parquet_dir: str = str(snakemake.params.parquet_dir)
	out_path: str = str(snakemake.output.matrix)
	start: int = int(snakemake.params.start)
	end: int = int(snakemake.params.end)
	gran: int = int(snakemake.params.gran)
	max_working_age: int = int(snakemake.params.max_working_age)

	log.info(
		"Loading MLER Parquet for batch [%d, %d] at observation month t_end=%d",
		start,
		end,
		end,
	)
	loader: MLERLoader = MLERLoader(parquet_dir)
	df_end: pd.DataFrame = loader.load_period(end)
	log.info("Loaded %d job relations at period %d", len(df_end), end)

	sector_series: pd.Series = loader.get_sector_column(gran, df_end)

	builder: BipartiteBuilder = BipartiteBuilder(
		df=df_end,
		granularity=gran,
		target_period=end,
		max_working_age=max_working_age,
	)
	states_df: pd.DataFrame = builder.extract_worker_states(sector_series)
	log.info(
		"Extracted %d primary working-age worker states across %d sectors",
		len(states_df),
		states_df["sector"].nunique() if not states_df.empty else 0,
	)

	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	states_df.to_parquet(out_path, index=False)
	log.info("Saved batch worker states to %s", out_path)


if __name__ == "__main__":
	main()
