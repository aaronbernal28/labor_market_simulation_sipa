"""
MLER data loader.

Loads the preprocessed year-partitioned MLER Parquet dataset and provides
filtered views by target period (e.g. batch end month t_end) and
sector-granularity level.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

# Mapping from granularity level -> column name.
GRANULARITY_COLUMN: dict[int, str] = {
	1: "letra",
	2: "r32",
	3: "r32",  # truncated to first digit after load
	4: "r34",
}


class MLERLoader:
	"""
	Loads and filters MLER Parquet data for a given target period or time batch
	and sector-granularity level.

	Parameters
	----------
	parquet_dir : str | Path
	    Path to the year-partitioned Parquet directory
	    (``data/processed/mler_parquet``).
	"""

	REQUIRED_COLS: ClassVar[list[str]] = [
		"id_trabajador",
		"id_relacion",
		"tiempo",
		"letra",
		"r32",
		"r34",
		"rem_tot",
		"fnacim",
		"pondera",
	]

	def __init__(self, parquet_dir: str | Path):
		self.parquet_dir: Path = Path(parquet_dir)

	def load_period(self, target_period: int) -> pd.DataFrame:
		"""
		Load rows for a specific YYYYMM period (e.g., t_end = 199612)
		using partition pruning and PyArrow pushdown filtering.
		"""
		year: int = target_period // 100
		year_path: Path = self.parquet_dir / f"year={year}"
		if not year_path.exists():
			return pd.DataFrame(columns=self.REQUIRED_COLS)

		df: pd.DataFrame = pd.read_parquet(
			year_path,
			filters=[("tiempo", "==", target_period)],
			columns=self.REQUIRED_COLS,
		)
		return df

	def filter_batch(self, start: int, end: int) -> pd.DataFrame:
		"""
		Return rows whose ``tiempo`` falls within [start, end] (YYYYMM).
		Reads only the relevant year partitions.
		"""
		start_year: int = start // 100
		end_year: int = end // 100
		frames: list[pd.DataFrame] = []
		for year in range(start_year, end_year + 1):
			year_path: Path = self.parquet_dir / f"year={year}"
			if year_path.exists():
				part: pd.DataFrame = pd.read_parquet(
					year_path,
					filters=[("tiempo", ">=", start), ("tiempo", "<=", end)],
					columns=self.REQUIRED_COLS,
				)
				frames.append(part)
		if not frames:
			return pd.DataFrame(columns=self.REQUIRED_COLS)
		return pd.concat(frames, ignore_index=True)

	@staticmethod
	def get_sector_column(granularity: int, df: pd.DataFrame) -> pd.Series:
		"""
		Extract the sector column for a given granularity level.

		For ``granularity=3``, the 2-digit column ``r32`` is truncated to
		its first digit (yielding a coarser classification).
		"""
		if granularity not in GRANULARITY_COLUMN:
			raise ValueError(
				f"Unknown granularity {granularity}. "
				f"Must be one of {list(GRANULARITY_COLUMN)}"
			)
		col: str = GRANULARITY_COLUMN[granularity]
		series: pd.Series = df[col].astype(str)
		if granularity == 3:
			series = series.str[:1]
		return series
