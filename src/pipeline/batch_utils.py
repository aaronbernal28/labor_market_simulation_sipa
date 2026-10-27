"""
Time batch partitioning utilities for the MLER pipeline.

Partitions the MLER time range (YYYYMM integers) into non-overlapping
batches of configurable width, used as Snakemake wildcards.
"""

from dataclasses import dataclass


@dataclass
class TimeBatch:
	"""Represents a single time partition window."""

	start: int  # YYYYMM
	end: int  # YYYYMM

	@property
	def batch_id(self) -> str:
		"""String identifier used in file paths / wildcards."""
		return f"{self.start}_{self.end}"

	def __repr__(self) -> str:
		return f"TimeBatch({self.start}–{self.end})"


class TimeBatchPartitioner:
	"""
	Partitions the MLER time range into non-overlapping batches
	of a fixed width (in months).

	Parameters
	----------
	start : int
	    First period in YYYYMM format (e.g. 199601).
	end : int
	    Last period in YYYYMM format (e.g. 202112).
	interval : int
	    Width of each batch in months (e.g. 12 for annual batches).

	Examples
	--------
	>>> p = TimeBatchPartitioner(start=199601, end=199812, interval=12)
	>>> p.batch_ids()
	['199601_199612', '199701_199812']
	"""

	def __init__(self, start: int, end: int, interval: int):
		self.start = start
		self.end = end
		self.interval = interval
		self._batches: list[TimeBatch] = []
		self._build()

	def _build(self) -> None:
		cursor = self.start
		while cursor <= self.end:
			batch_end = self._add_months(cursor, self.interval - 1)
			batch_end = min(batch_end, self.end)
			self._batches.append(TimeBatch(start=cursor, end=batch_end))
			cursor = self._add_months(batch_end, 1)

	@staticmethod
	def _add_months(yyyymm: int, months: int) -> int:
		"""Add ``months`` to a YYYYMM integer, handling year rollovers."""
		year, month = divmod(yyyymm, 100)
		month += months
		year += (month - 1) // 12
		month = (month - 1) % 12 + 1
		return year * 100 + month

	def batches(self) -> list[TimeBatch]:
		"""Return the ordered list of TimeBatch objects."""
		return list(self._batches)

	def batch_ids(self) -> list[str]:
		"""Return batch identifiers suitable for Snakemake wildcards."""
		return [b.batch_id for b in self._batches]

	def consecutive_pairs(self) -> list[tuple]:
		"""
		Return pairs of (batch_t, batch_t+1) for computing transitions.

		The transition matrix for batch *t* requires data from both
		batch *t* and batch *t+1*.
		"""
		return list(zip(self._batches[:-1], self._batches[1:]))
