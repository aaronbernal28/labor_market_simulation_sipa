"""
Script: preprocess_data.py
Called by: workflow/rules/extract.smk -> rule preprocess_mler

Reads the raw MLER CSV in chunks, filters out records with null fnacim,
writes year-partitioned Parquet files, and builds the worker registry
(id_trabajador, first_period, fnacim, pondera).
"""

import logging
import shutil
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


def preprocess_mler(
	csv_path: str | Path,
	out_dir: str | Path,
	registry_path: str | Path,
	chunksize: int = 1_000_000,
) -> None:
	csv_path = Path(csv_path)
	out_dir = Path(out_dir)
	registry_path = Path(registry_path)

	if out_dir.exists():
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	registry_path.parent.mkdir(parents=True, exist_ok=True)

	log.info("Starting preprocessing of %s", csv_path)

	# Keep track of worker registry across chunks
	worker_chunks = []
	total_rows = 0
	kept_rows = 0

	usecols = [
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

	dtype = {
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
		csv_path,
		usecols=usecols,
		dtype=dtype,
		chunksize=chunksize,
	)

	for chunk_idx, chunk in enumerate(reader):
		n_raw = len(chunk)
		total_rows += n_raw

		# Filter out null fnacim
		valid = chunk.dropna(subset=["fnacim"]).copy()
		valid["fnacim"] = valid["fnacim"].astype("int16")
		valid["year"] = (valid["tiempo"] // 100).astype("int16")
		kept_rows += len(valid)

		# Write partitioned by year
		for year, group in valid.groupby("year"):
			year_dir = out_dir / f"year={year}"
			year_dir.mkdir(parents=True, exist_ok=True)
			part_file = year_dir / f"part_{chunk_idx:04d}.parquet"
			group.drop(columns=["year"]).to_parquet(part_file, index=False)

		# Aggregate worker metadata for this chunk
		w_agg = (
			valid.groupby("id_trabajador", as_index=False)
			.agg(
				first_period=("tiempo", "min"),
				fnacim=("fnacim", "first"),
				pondera=("pondera", "first"),
			)
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
	full_workers = pd.concat(worker_chunks, ignore_index=True)
	registry = (
		full_workers.groupby("id_trabajador", as_index=False)
		.agg(
			first_period=("first_period", "min"),
			fnacim=("fnacim", "first"),
			pondera=("pondera", "first"),
		)
		.sort_values("id_trabajador")
		.reset_index(drop=True)
	)

	registry.to_parquet(registry_path, index=False)
	log.info(
		"Saved worker registry with %d unique workers to %s",
		len(registry),
		registry_path,
	)


def main():
	csv_path = _sm.input.csv if _sm else sys.argv[1]
	out_dir = _sm.params.out_dir if _sm else sys.argv[2]
	registry_path = _sm.output.registry if _sm else sys.argv[3]

	preprocess_mler(
		csv_path=csv_path,
		out_dir=out_dir,
		registry_path=registry_path,
	)


if __name__ == "__main__":
	main()
