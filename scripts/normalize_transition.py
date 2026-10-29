"""
Script: normalize_transition.py
Called by: workflow/rules/transition.smk -> rule normalize_transition

Row-normalizes a computed Hidalgo proximity matrix into fixed transition
probabilities suitable for agent random sampling in the Mesa ABM.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.probability_normalizer import normalize_rows

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
	matrix_path: str = str(snakemake.input.matrix)
	out_path: str = str(snakemake.output.probs)

	log.info("Loading Hidalgo proximity matrix from %s", matrix_path)
	matrix_df: pd.DataFrame = pd.read_parquet(matrix_path)

	log.info(
		"Row-normalizing %d x %d proximity matrix into transition probabilities",
		matrix_df.shape[0],
		matrix_df.shape[1],
	)
	probs_df: pd.DataFrame = normalize_rows(matrix_df)

	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	probs_df.to_csv(out_path, index=True)
	log.info("Saved normalized transition probabilities to %s", out_path)


if __name__ == "__main__":
	main()
