"""
Script: export_transition_csv.py
Called by: workflow/rules/transition.smk -> rule export_transition_csv

Reads a computed transition matrix Parquet file, maps raw sector codes
to human-readable descriptions from data/anexo_estadistico_y_descriptores.xlsx,
and exports a formatted CSV matrix for inspection and visualization.
"""

import logging
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


def load_sector_labels(excel_path: str | Path, granularity: int) -> dict[str, str]:
	"""Load sector code -> human-readable label mapping from Excel descriptors."""
	labels: dict[str, str] = {"unemployed": "Desempleo / Inactividad"}

	if granularity == 1:
		df_l: pd.DataFrame = pd.read_excel(
			excel_path, sheet_name="DESC_LETRA", skiprows=2, engine="openpyxl"
		)
		df_l.columns = ["code", "letter", "desc"]
		for _, row in df_l.iterrows():
			if pd.notna(row["code"]):
				c_str: str = str(int(row["code"]))
				letter: str = (
					str(row["letter"]).strip() if pd.notna(row["letter"]) else ""
				)
				desc: str = str(row["desc"]).strip() if pd.notna(row["desc"]) else ""
				labels[c_str] = f"{letter} - {desc}" if letter else desc

	elif granularity == 2:
		df_r: pd.DataFrame = pd.read_excel(
			excel_path, sheet_name="DESC_R32", skiprows=2, engine="openpyxl"
		)
		df_r.columns = ["code", "desc"]
		for _, row in df_r.iterrows():
			if pd.notna(row["code"]):
				c_str = str(int(row["code"]))
				desc = str(row["desc"]).strip() if pd.notna(row["desc"]) else ""
				labels[c_str] = f"{c_str} - {desc}"

	elif granularity == 4:
		df_4: pd.DataFrame = pd.read_excel(
			excel_path, sheet_name="DESC_R34", skiprows=2, engine="openpyxl"
		)
		df_4.columns = ["code", "desc"]
		for _, row in df_4.iterrows():
			if pd.notna(row["code"]):
				c_str = str(int(row["code"]))
				desc = str(row["desc"]).strip() if pd.notna(row["desc"]) else ""
				labels[c_str] = f"{c_str} - {desc}"

	elif granularity == 3:
		for i in range(10):
			labels[str(i)] = f"Grupo {i}"

	return labels


def main() -> None:
	parquet_path: str = str(snakemake.input.matrix)
	excel_path: str = str(snakemake.input.descriptors)
	csv_path: str = str(snakemake.output.csv)
	gran: int = int(snakemake.wildcards.gran)

	log.info("Loading Parquet matrix from %s", parquet_path)
	matrix_df: pd.DataFrame = pd.read_parquet(parquet_path)

	log.info("Loading sector descriptors (granularity=%d) from %s", gran, excel_path)
	label_map: dict[str, str] = load_sector_labels(excel_path, gran)

	# Relabel rows and columns with human-readable descriptions
	matrix_df.index = [label_map.get(str(idx), str(idx)) for idx in matrix_df.index]
	matrix_df.columns = [label_map.get(str(col), str(col)) for col in matrix_df.columns]
	matrix_df.index.name = "Sector de origen"
	matrix_df.columns.name = "Sector de destino"

	Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
	matrix_df.to_csv(csv_path, index=True)
	log.info(
		"Saved %d x %d labeled matrix to %s",
		matrix_df.shape[0],
		matrix_df.shape[1],
		csv_path,
	)


if __name__ == "__main__":
	main()
