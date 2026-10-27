"""
Bipartite & Worker Batch State builder.

Evaluates worker sector states at the batch end month (t_end),
applying the working-age retirement cutoff (Age <= max_working_age)
and resolving multiple concurrent employment relations via highest
gross remuneration (max rem_tot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class BipartiteBuilder:
	"""
	Builds worker sector assignments at a batch observation point (t_end)
	and provides both a tabular state representation and a sparse
	worker x sector occurrence matrix.

	Parameters
	----------
	df : pd.DataFrame
	    Filtered MLER records at month ``t_end``, containing columns
	    ``id_trabajador``, ``rem_tot``, ``fnacim``, ``pondera``, and
	    the sector column.
	granularity : int
	    Sector granularity level (1=letra, 2=r32, 3=r32_1dig, 4=r34).
	target_period : int
	    The observation period YYYYMM (e.g. 199612), used to compute worker age.
	max_working_age : int
	    Maximum working age (default 65). Workers older than this are excluded.
	"""

	def __init__(
		self,
		df: pd.DataFrame,
		granularity: int,
		target_period: int,
		max_working_age: int = 65,
	):
		self.df = df
		self.granularity = granularity
		self.target_period = target_period
		self.max_working_age = max_working_age

	def extract_worker_states(self, sector_series: pd.Series) -> pd.DataFrame:
		"""
		Resolve each working-age worker's primary sector at ``target_period``.

		1. Filter out workers with ``Year(target_period) - fnacim > max_working_age``.
		2. If a worker holds multiple concurrent jobs in ``target_period``,
		   select the job with maximum ``rem_tot``.
		3. Return a DataFrame with columns ``['id_trabajador', 'sector', 'pondera']``.
		"""
		if self.df.empty:
			return pd.DataFrame(columns=["id_trabajador", "sector", "pondera"])

		work = self.df[["id_trabajador", "rem_tot", "fnacim", "pondera"]].copy()
		work["sector"] = sector_series.astype(str).values

		obs_year = self.target_period // 100
		age = obs_year - work["fnacim"]
		work = work[age <= self.max_working_age].copy()

		if work.empty:
			return pd.DataFrame(columns=["id_trabajador", "sector", "pondera"])

		# Sort by id_trabajador asc, rem_tot desc so drop_duplicates keeps max rem_tot
		work = work.sort_values(
			by=["id_trabajador", "rem_tot"], ascending=[True, False]
		)
		primary = (
			work.drop_duplicates(subset=["id_trabajador"], keep="first")[
				["id_trabajador", "sector", "pondera"]
			]
			.reset_index(drop=True)
		)
		return primary

	@staticmethod
	def to_sparse_bipartite(
		states_df: pd.DataFrame,
		worker_index: list | np.ndarray,
		sector_labels: list[str],
		weighted: bool = True,
	) -> csr_matrix:
		"""
		Construct a sparse (Workers x Sectors) CSR matrix aligned to
		``worker_index`` and ``sector_labels``.
		"""
		n_workers = len(worker_index)
		n_sectors = len(sector_labels)
		if states_df.empty or n_workers == 0:
			return csr_matrix((n_workers, n_sectors), dtype=np.float64)

		w_map = pd.Series(np.arange(n_workers), index=worker_index)
		s_map = {s: i for i, s in enumerate(sector_labels)}

		valid = states_df[
			states_df["id_trabajador"].isin(w_map.index)
			& states_df["sector"].isin(s_map)
		]
		rows = w_map.loc[valid["id_trabajador"]].values
		cols = valid["sector"].map(s_map).values
		vals = (
			valid["pondera"].astype(np.float64).values
			if weighted
			else np.ones(len(valid), dtype=np.float64)
		)

		return csr_matrix((vals, (rows, cols)), shape=(n_workers, n_sectors))
