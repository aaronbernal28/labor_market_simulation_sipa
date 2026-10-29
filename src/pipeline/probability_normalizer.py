r"""
Probability Normalizer.

Converts non-negative matrix scores (such as Hidalgo proximity matrices)
into row-stochastic transition probability matrices suitable for agent sampling.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def normalize_rows(matrix: pd.DataFrame) -> pd.DataFrame:
	r"""
	Row-normalize a non-negative (N+1) x (N+1) matrix into transition probabilities.

	Notes
	-----
	# FIXME: Applied to Hidalgo proximity \phi = C / \max(N_t, N_{t+1}); the result is a
	# proximity-weighted approximation, not the empirical P(s_j | s_i).
	# FIXME: Rows with zero total mass get P_{i, i} = 1.0 (agent stays in current sector).

	Parameters
	----------
	matrix : pd.DataFrame
		Square (N+1) x (N+1) DataFrame of non-negative scores.

	Returns
	-------
	pd.DataFrame
		Row-stochastic matrix where each row sums to 1.0.
	"""
	if matrix.empty:
		return matrix.copy()

	values: npt.NDArray[np.float64] = matrix.values.astype(np.float64)
	row_sums: npt.NDArray[np.float64] = values.sum(axis=1, keepdims=True)

	with np.errstate(divide="ignore", invalid="ignore"):
		normalized: npt.NDArray[np.float64] = np.where(
			row_sums > 0.0, values / row_sums, 0.0
		)

	# Handle zero-sum rows: agent remains in sector (absorbing state)
	zero_rows: npt.NDArray[np.bool_] = row_sums.squeeze(axis=1) == 0.0
	if np.any(zero_rows):
		zero_indices: npt.NDArray[np.int64] = np.where(zero_rows)[0]
		for idx in zero_indices:
			normalized[idx, idx] = 1.0

	# Re-normalize slightly in case of floating point rounding drift
	final_sums: npt.NDArray[np.float64] = normalized.sum(axis=1, keepdims=True)
	with np.errstate(divide="ignore", invalid="ignore"):
		normalized = np.where(final_sums > 0.0, normalized / final_sums, normalized)

	result: pd.DataFrame = pd.DataFrame(
		normalized, index=matrix.index, columns=matrix.columns
	)
	result.index.name = matrix.index.name
	result.columns.name = matrix.columns.name
	return result
