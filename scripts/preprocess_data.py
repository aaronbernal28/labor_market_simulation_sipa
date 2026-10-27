"""
Script: preprocess_data.py
Called by: workflow/rules/extract.smk -> rule preprocess_mler

Reads the raw MLER CSV in chunks, filters out records with null fnacim,
writes year-partitioned Parquet files, and builds the worker registry
(id_trabajador, first_period, fnacim, pondera).
"""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

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


def preprocess_mler(
	csv_path: str | Path,
	out_dir: str | Path,
	registry_path: str | Path,
	chunksize: int = 1_000_000,
) -> None:
	csv_path_obj: Path = Path(csv_path)
	out_dir_obj: Path = Path(out_dir)
	registry_path_obj: Path = Path(registry_path)

	if out_dir_obj.exists():
		shutil.rmtree(out_dir_obj)
	out_dir_obj.mkdir(parents=True, exist_ok=True)
	registry_path_obj.parent.mkdir(parents=True, exist_ok=True)

	log.info("Starting preprocessing of %s", csv_path_obj)

	# Keep track of worker registry across chunks
	worker_chunks: list[pd.DataFrame] = []
	total_rows: int = 0
	kept_rows: int = 0

	usecols: list[str] = [
		"id_trabajador",
		"id_relacion",
		"tiempo",
		"rem_tot",
		"r34",
		"r32",
		"letra",
		"fnacim",
		"sexo",
		"pondera",
	]

	dtype: dict[str, str] = {
		"id_trabajador": "int64",
		"id_relacion": "int32",
		"tiempo": "int32",
		"rem_tot": "float32",
		"r34": "str",
		"r32": "str",
		"letra": "str",
		"fnacim": "float64",  # float to allow NaNs before filtering
		"sexo": "int8",
		"pondera": "float32",
	}

	reader = pd.read_csv(
		csv_path_obj,
		usecols=usecols,
		dtype=dtype,
		chunksize=chunksize,
	)

	for chunk_idx, chunk in enumerate(reader):
		n_raw: int = len(chunk)
		total_rows += n_raw

		# Filter out null fnacim
		valid: pd.DataFrame = chunk.dropna(subset=["fnacim"]).copy()
		valid["fnacim"] = valid["fnacim"].astype("int16")
		valid["year"] = (valid["tiempo"] // 100).astype("int16")
		kept_rows += len(valid)

		# Write partitioned by year
		for year, group in valid.groupby("year"):
			year_dir: Path = out_dir_obj / f"year={year}"
			year_dir.mkdir(parents=True, exist_ok=True)
			part_file: Path = year_dir / f"part_{chunk_idx:04d}.parquet"
			group.drop(columns=["year"]).to_parquet(part_file, index=False)

		# Aggregate worker metadata for this chunk
		w_agg: pd.DataFrame = valid.groupby("id_trabajador", as_index=False).agg(
			first_period=("tiempo", "min"),
			fnacim=("fnacim", "first"),
			pondera=("pondera", "first"),
		)
		worker_chunks.append(w_agg)

		log.info(
			"Processed chunk %d: %d/%d rows kept (cumulative %d/%d)",
			chunk_idx,
			len(valid),
			n_raw,
			kept_rows,
			total_rows,
		)

	log.info("Consolidating worker registry from %d chunks", len(worker_chunks))
	full_workers: pd.DataFrame = pd.concat(worker_chunks, ignore_index=True)
	registry: pd.DataFrame = (
		full_workers.groupby("id_trabajador", as_index=False)
		.agg(
			first_period=("first_period", "min"),
			fnacim=("fnacim", "first"),
			pondera=("pondera", "first"),
		)
		.sort_values("id_trabajador")
		.reset_index(drop=True)
	)

	registry.to_parquet(registry_path_obj, index=False)
	log.info(
		"Saved worker registry with %d unique workers to %s",
		len(registry),
		registry_path_obj,
	)


def main() -> None:
	csv_path: str = str(snakemake.input.csv)
	out_dir: str = str(snakemake.params.out_dir)
	registry_path: str = str(snakemake.output.registry)

	preprocess_mler(
		csv_path=csv_path,
		out_dir=out_dir,
		registry_path=registry_path,
	)


if __name__ == "__main__":
	main()
