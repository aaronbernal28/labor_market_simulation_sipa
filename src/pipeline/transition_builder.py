r"""
Transition and Hidalgo Proximity Matrix Builder.

Computes empirical flow counts C_{t, s_i, s_j}, transition probabilities
P_{t+1}(s_j | s_i), and the Hidalgo proximity matrix \phi_{t, s_i, s_j}
over the state space S = {s_0, s_1, ..., s_N}, where s_0 = 'unemployed'.

See docs/definitions.md for the full mathematical formulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


class BipartiteToTransition:
	"""
	Convert consecutive batch worker state assignments and the working-age
	cumulative worker registry into an (N+1) x (N+1) transition or
	Hidalgo proximity matrix.

	Parameters
	----------
	states_t : pd.DataFrame
	    Worker states at batch t (columns: ``id_trabajador``, ``sector``, ``pondera``).
	states_t1 : pd.DataFrame
	    Worker states at batch t+1 (columns: ``id_trabajador``, ``sector``, ``pondera``).
	active_universe : pd.DataFrame
	    Cumulative working-age worker pool up to batch t+1
	    (columns: ``id_trabajador``, ``pondera``).
	all_sectors : list[str] | None
	    Explicit list of economic sector labels. If None, inferred from the
	    union of sectors in ``states_t`` and ``states_t1``.
	unemployed_label : str
	    Label used for state s_0 (default ``"unemployed"``).
	"""

	def __init__(
		self,
		states_t: pd.DataFrame,
		states_t1: pd.DataFrame,
		active_universe: pd.DataFrame,
		all_sectors: list[str] | None = None,
		unemployed_label: str = "unemployed",
	):
		self.unemployed_label = unemployed_label
		self.states_t = (
			states_t.set_index("id_trabajador")["sector"].astype(str)
			if not states_t.empty
			else pd.Series(dtype=str)
		)
		self.states_t1 = (
			states_t1.set_index("id_trabajador")["sector"].astype(str)
			if not states_t1.empty
			else pd.Series(dtype=str)
		)
		self.universe = active_universe.drop_duplicates(
			subset=["id_trabajador"]
		).set_index("id_trabajador")

		if all_sectors is None:
			sectors_union = set(self.states_t.unique()) | set(self.states_t1.unique())
			sectors_union.discard(self.unemployed_label)
			self.all_sectors = sorted(sectors_union, key=self._sector_sort_key)
		else:
			self.all_sectors = sorted(
				[str(s) for s in all_sectors if str(s) != self.unemployed_label],
				key=self._sector_sort_key,
			)

		self.labels = [self.unemployed_label] + self.all_sectors

	@staticmethod
	def _sector_sort_key(val: str) -> tuple[int, int | str]:
		"""Sort numeric sector codes numerically first, then lexicographically."""
		try:
			return (0, int(val))
		except ValueError:
			return (1, val)

	def compute_weighted_flows(self) -> pd.DataFrame:
		"""
		Compute the weighted flow count matrix C_{t, s_i, s_j} = \\sum_u \\omega_u.

		Returns
		-------
		pd.DataFrame
		    (N+1) x (N+1) DataFrame with index ``sector_from`` and
		    columns ``sector_to``.
		"""
		if self.universe.empty:
			empty = pd.DataFrame(
				0.0, index=self.labels, columns=self.labels, dtype=np.float64
			)
			empty.index.name = "sector_from"
			empty.columns.name = "sector_to"
			return empty

		df = pd.DataFrame(index=self.universe.index)
		df["pondera"] = self.universe["pondera"].astype(np.float64)
		df["sector_from"] = self.states_t.reindex(
			df.index, fill_value=self.unemployed_label
		)
		df["sector_to"] = self.states_t1.reindex(
			df.index, fill_value=self.unemployed_label
		)

		C = (
			df.groupby(["sector_from", "sector_to"], observed=True)["pondera"]
			.sum()
			.unstack(fill_value=0.0)
			.reindex(index=self.labels, columns=self.labels, fill_value=0.0)
		)
		C.index.name = "sector_from"
		C.columns.name = "sector_to"
		return C

	def compute_hidalgo_proximity(self) -> pd.DataFrame:
		"""
		Compute the Hidalgo proximity matrix \\Phi_t:
		\\phi_{t, s_i, s_j} = C_{t, s_i, s_j} / \\max( N_t(s_i), N_{t+1}(s_j) )
		"""
		C_df: pd.DataFrame = self.compute_weighted_flows()
		C: npt.NDArray[np.float64] = C_df.values.astype(np.float64)

		N_t: npt.NDArray[np.float64] = C.sum(axis=1)  # shape: (N+1,)
		N_t1: npt.NDArray[np.float64] = C.sum(axis=0)  # shape: (N+1,)

		D: npt.NDArray[np.float64] = np.maximum(N_t[:, np.newaxis], N_t1[np.newaxis, :])

		with np.errstate(divide="ignore", invalid="ignore"):
			Phi: npt.NDArray[np.float64] = np.where(D > 0.0, C / D, 0.0)

		res: pd.DataFrame = pd.DataFrame(Phi, index=self.labels, columns=self.labels)
		res.index.name = "sector_from"
		res.columns.name = "sector_to"
		return res

	def compute_row_stochastic(self) -> pd.DataFrame:
		"""
		Compute the forward Markov transition probability matrix P_{t+1}(s_j | s_i):
		T_{i, j} = C_{t, s_i, s_j} / N_t(s_i)
		"""
		C_df: pd.DataFrame = self.compute_weighted_flows()
		C: npt.NDArray[np.float64] = C_df.values.astype(np.float64)
		row_sums: npt.NDArray[np.float64] = C.sum(axis=1, keepdims=True)

		with np.errstate(divide="ignore", invalid="ignore"):
			T: npt.NDArray[np.float64] = np.where(row_sums > 0.0, C / row_sums, 0.0)

		res: pd.DataFrame = pd.DataFrame(T, index=self.labels, columns=self.labels)
		res.index.name = "sector_from"
		res.columns.name = "sector_to"
		return res

	def compute(self, metric: str = "hidalgo") -> pd.DataFrame:
		"""
		Dispatch computation based on ``metric`` ('hidalgo' or 'row_stochastic').
		"""
		if metric == "hidalgo":
			return self.compute_hidalgo_proximity()
		if metric == "row_stochastic":
			return self.compute_row_stochastic()
		raise ValueError(
			f"Unknown transition metric '{metric}'. "
			"Expected 'hidalgo' or 'row_stochastic'."
		)
