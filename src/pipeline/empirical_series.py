r"""
Empirical Labor Market Series & Fluctuation Metrics.

Computes the macroeconomic state vector \mathbf{S}_t, Shannon entropy H_t,
net sectoral variations \Delta x_t, and the Lilien sectoral reallocation index L_t
across the empirical SIPA panel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def compute_period_headcounts(
	states: pd.DataFrame,
	active_universe: pd.DataFrame,
	unemployed_label: str = "unemployed",
) -> pd.Series:
	r"""
	Compute weighted sectoral headcounts N_t(s_i) for a single period.

	Workers in ``active_universe`` who have a recorded formal job in ``states``
	are assigned to that sector. Active workers absent from ``states`` are
	assigned to the ``unemployed_label`` (state s_0).

	Parameters
	----------
	states : pd.DataFrame
		Worker states for period t with columns ``id_trabajador``, ``sector``,
		and ``pondera``.
	active_universe : pd.DataFrame
		Active working-age worker registry up to period t with columns
		``id_trabajador`` and ``pondera``.
	unemployed_label : str
		Label used for state s_0 (default ``"unemployed"``).

	Returns
	-------
	pd.Series
		Weighted headcounts indexed by sector, including ``unemployed_label``.
	"""
	if active_universe.empty:
		return pd.Series(dtype=np.float64)

	# Deduplicate universe by worker ID
	universe: pd.DataFrame = active_universe.drop_duplicates(
		subset=["id_trabajador"]
	).set_index("id_trabajador")

	worker_sector: pd.Series = (
		states.drop_duplicates(subset=["id_trabajador"])
		.set_index("id_trabajador")["sector"]
		.astype(str)
		if not states.empty
		else pd.Series(dtype=str)
	)

	df: pd.DataFrame = pd.DataFrame(index=universe.index)
	df["pondera"] = universe["pondera"].astype(np.float64)
	df["sector"] = worker_sector.reindex(df.index, fill_value=unemployed_label)

	headcounts: pd.Series = (
		df.groupby("sector", observed=True)["pondera"].sum().astype(np.float64)
	)

	# Ensure unemployed label exists even if count is zero
	if unemployed_label not in headcounts.index:
		headcounts[unemployed_label] = 0.0

	return headcounts


def compute_shares(headcounts: pd.Series) -> pd.Series:
	r"""
	Compute relative sectoral shares x_t(s_i) = N_t(s_i) / M_t.

	Parameters
	----------
	headcounts : pd.Series
		Sectoral headcounts including unemployed.

	Returns
	-------
	pd.Series
		Shares summing to 1.0 (empty series if total mass is zero).
	"""
	total: float = float(headcounts.sum())
	if total <= 0.0:
		return pd.Series(0.0, index=headcounts.index, dtype=np.float64)
	return (headcounts / total).astype(np.float64)


def compute_shannon_entropy(shares: pd.Series) -> float:
	r"""
	Compute the Shannon entropy of the labor distribution:
	H_t = - \sum_{i=0}^N x_t(s_i) \log_2 x_t(s_i)

	Parameters
	----------
	shares : pd.Series
		Sectoral shares summing to 1.0.

	Returns
	-------
	float
		Entropy in bits (0.0 if empty or degenerate).
	"""
	vals: npt.NDArray[np.float64] = shares.to_numpy(dtype=np.float64)
	positive: npt.NDArray[np.float64] = vals[vals > 0.0]
	if len(positive) == 0:
		return 0.0
	return float(-np.sum(positive * np.log2(positive)))


def compute_lilien_index(
	headcounts_t: pd.Series,
	headcounts_t1: pd.Series,
	unemployed_label: str = "unemployed",
) -> float:
	r"""
	Compute the Lilien Sectoral Reallocation Index:
	L_t = \sqrt{ \sum_{i=1}^N x_t(s_i) ( \Delta \ln N_t(s_i) - \Delta \ln E_t )^2 }

	Parameters
	----------
	headcounts_t : pd.Series
		Headcounts at period t.
	headcounts_t1 : pd.Series
		Headcounts at period t+1.
	unemployed_label : str
		Label used for unemployed state s_0 (excluded from formal sectors).

	Returns
	-------
	float
		Lilien reallocation index (0.0 if no formal employment exists).
	"""
	# Filter out unemployment to isolate formal employment
	h_t: pd.Series = headcounts_t[headcounts_t.index != unemployed_label].astype(
		np.float64
	)
	h_t1: pd.Series = headcounts_t1[headcounts_t1.index != unemployed_label].astype(
		np.float64
	)

	e_t: float = float(h_t.sum())
	e_t1: float = float(h_t1.sum())

	if e_t <= 0.0 or e_t1 <= 0.0:
		return 0.0

	# Align formal sectors across both periods
	all_formal: list[str] = sorted(set(h_t.index) | set(h_t1.index))
	h_t_aligned: pd.Series = h_t.reindex(all_formal, fill_value=0.0)
	h_t1_aligned: pd.Series = h_t1.reindex(all_formal, fill_value=0.0)

	delta_ln_e: float = float(np.log(e_t1) - np.log(e_t))

	# Restrict summation to sectors with non-zero employment in both periods
	valid: pd.Series = (h_t_aligned > 0.0) & (h_t1_aligned > 0.0)
	if not valid.any():
		return 0.0

	n0: npt.NDArray[np.float64] = h_t_aligned[valid].to_numpy()
	n1: npt.NDArray[np.float64] = h_t1_aligned[valid].to_numpy()
	w: npt.NDArray[np.float64] = n0 / e_t

	delta_ln_n: npt.NDArray[np.float64] = np.log(n1) - np.log(n0)
	diff: npt.NDArray[np.float64] = delta_ln_n - delta_ln_e
	variance: float = float(np.sum(w * (diff**2)))

	return float(np.sqrt(max(0.0, variance)))


def build_empirical_series(
	headcounts_by_period: dict[int, pd.Series],
	unemployed_label: str = "unemployed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
	r"""
	Build macro and sectoral longitudinal tables across all observed periods.

	Parameters
	----------
	headcounts_by_period : dict[int, pd.Series]
		Mapping from period (YYYYMM) to headcount Series.
	unemployed_label : str
		Label used for state s_0 (default ``"unemployed"``).

	Returns
	-------
	tuple[pd.DataFrame, pd.DataFrame]
		``(macro_df, sectoral_df)``:
		- ``macro_df``: columns [period, active_pool, formal_employed, unemployed,
		  unemployment_rate, entropy, lilien_index]
		- ``sectoral_df``: columns [period, sector, headcount, share, delta_share]
	"""
	periods: list[int] = sorted(headcounts_by_period.keys())
	if not periods:
		empty_macro = pd.DataFrame(
			columns=[
				"period",
				"active_pool",
				"formal_employed",
				"unemployed",
				"unemployment_rate",
				"entropy",
				"lilien_index",
			]
		)
		empty_sectoral = pd.DataFrame(
			columns=["period", "sector", "headcount", "share", "delta_share"]
		)
		return empty_macro, empty_sectoral

	macro_records: list[dict[str, int | float | None]] = []
	sectoral_records: list[dict[str, int | str | float | None]] = []

	prev_period: int | None = None
	prev_shares: dict[str, float] = {}

	for period in periods:
		counts: pd.Series = headcounts_by_period[period]
		shares: pd.Series = compute_shares(counts)

		active_pool: float = float(counts.sum())
		unemployed: float = float(counts.get(unemployed_label, 0.0))
		formal_employed: float = float(active_pool - unemployed)
		unemployment_rate: float = (
			float(unemployed / active_pool) if active_pool > 0.0 else 0.0
		)
		entropy: float = compute_shannon_entropy(shares)

		lilien: float | None = None
		if prev_period is not None:
			lilien = compute_lilien_index(
				headcounts_t=headcounts_by_period[prev_period],
				headcounts_t1=counts,
				unemployed_label=unemployed_label,
			)

		macro_records.append(
			{
				"period": period,
				"active_pool": active_pool,
				"formal_employed": formal_employed,
				"unemployed": unemployed,
				"unemployment_rate": unemployment_rate,
				"entropy": entropy,
				"lilien_index": lilien,
			}
		)

		current_shares_dict: dict[str, float] = shares.to_dict()
		for sector, count in counts.items():
			sec_str: str = str(sector)
			share: float = current_shares_dict.get(sec_str, 0.0)
			delta_share: float | None = (
				share - prev_shares[sec_str] if sec_str in prev_shares else None
			)

			sectoral_records.append(
				{
					"period": period,
					"sector": sec_str,
					"headcount": float(count),
					"share": share,
					"delta_share": delta_share,
				}
			)

		prev_period = period
		prev_shares = current_shares_dict

	macro_df: pd.DataFrame = pd.DataFrame(macro_records)
	sectoral_df: pd.DataFrame = pd.DataFrame(sectoral_records)

	return macro_df, sectoral_df
