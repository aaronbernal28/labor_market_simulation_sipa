"""
Script: build_bipartite.py
Called by: workflow/rules/bipartite.smk -> rule build_bipartite

Loads the partitioned MLER Parquet slice at a given period (YYYYMM),
applies working-age filtering and primary job selection (max rem_tot),
and saves the resolved worker states to Parquet.
"""

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
	period: int = int(snakemake.params.period)
	gran: int = int(snakemake.params.gran)
	max_working_age: int = int(snakemake.params.max_working_age)

	log.info("Loading MLER Parquet at observation period=%d", period)
	loader: MLERLoader = MLERLoader(parquet_dir)
	df_period: pd.DataFrame = loader.load_period(period)
	log.info("Loaded %d job relations at period %d", len(df_period), period)

	sector_series: pd.Series = loader.get_sector_column(gran, df_period)

	builder: BipartiteBuilder = BipartiteBuilder(
		df=df_period,
		granularity=gran,
		target_period=period,
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
	log.info("Saved period worker states to %s", out_path)


if __name__ == "__main__":
	main()
