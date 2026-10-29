r"""
Initial State Perturbations for Agent Populations.

Constructs baseline and experimental initial agent populations:
- Baseline: Empirical shares via largest-remainder quota allocation.
- Experiment A: Mass layoff (20% of formal workers moved to unemployed).
- Experiment B: Sectoral contraction (50% of target sector redistributed uniformly).
- Experiment C: Uniform initial state across all sectors.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_baseline_sectors(shares: pd.Series, n_agents: int) -> list[str]:
	r"""
	Deterministically allocate ``n_agents`` to sectors using largest remainder.

	Ensures the initial agent distribution matches empirical shares without
	sampling variance at step 0, and that the sum of agent counts is exactly
	``n_agents``.

	Parameters
	----------
	shares : pd.Series
		Sectoral shares summing to 1.0.
	n_agents : int
		Total simulated agent count.

	Returns
	-------
	list[str]
		List of sector labels of length ``n_agents``.
	"""
	if n_agents <= 0 or shares.empty:
		return []

	clean_shares: pd.Series = shares[shares > 0.0]
	labels: list[str] = [str(s) for s in clean_shares.index]
	probs: npt.NDArray[np.float64] = clean_shares.to_numpy(dtype=np.float64)

	# Largest remainder method (Hamilton-Hare quota)
	quotas: npt.NDArray[np.float64] = probs * n_agents
	floor_counts: npt.NDArray[np.int64] = np.floor(quotas).astype(np.int64)
	remainder: int = n_agents - int(np.sum(floor_counts))

	if remainder > 0:
		fractions: npt.NDArray[np.float64] = quotas - floor_counts
		top_indices: npt.NDArray[np.int64] = np.argsort(-fractions)[:remainder]
		floor_counts[top_indices] += 1

	sectors: list[str] = []
	for label, count in zip(labels, floor_counts, strict=True):
		sectors.extend([label] * int(count))

	return sectors


def apply_mass_layoff(
	sectors: list[str],
	fraction: float = 0.20,
	rng: random.Random | None = None,
	unemployed_label: str = "unemployed",
) -> list[str]:
	r"""
	Experiment A: Displace a fraction of formal employed workers to unemployment.

	Parameters
	----------
	sectors : list[str]
		Initial agent sectors.
	fraction : float
		Proportion of formally employed workers to displace (default 0.20).
	rng : random.Random | None
		Seeded random instance for reproducible sampling.
	unemployed_label : str
		Label used for state s_0 (default ``"unemployed"``).

	Returns
	-------
	list[str]
		New list of sectors with layoffs applied.
	"""
	r: random.Random = rng if rng is not None else random.Random()
	result: list[str] = list(sectors)

	employed_indices: list[int] = [
		i for i, s in enumerate(result) if s != unemployed_label
	]
	n_layoffs: int = round(fraction * len(employed_indices))

	if n_layoffs > 0 and len(employed_indices) >= n_layoffs:
		chosen_indices: list[int] = r.sample(employed_indices, n_layoffs)
		for idx in chosen_indices:
			result[idx] = unemployed_label

	return result


def apply_sector_contraction(
	sectors: list[str],
	target: str,
	fraction: float = 0.50,
	rng: random.Random | None = None,
	unemployed_label: str = "unemployed",
) -> list[str]:
	r"""
	Experiment B: Contract target sector and redistribute displaced workers.

	Parameters
	----------
	sectors : list[str]
		Initial agent sectors.
	target : str
		Target sector to contract.
	fraction : float
		Contraction fraction (default 0.50).
	rng : random.Random | None
		Seeded random instance for reproducible sampling.
	unemployed_label : str
		Label used for state s_0.

	Notes
	-----
	# FIXME: Displaced workers are redistributed uniformly across the other formal sectors;
	# refine with Hidalgo proximity-weighted redistribution in future iterations.

	Returns
	-------
	list[str]
		New list of sectors with target sector contracted.
	"""
	r: random.Random = rng if rng is not None else random.Random()
	result: list[str] = list(sectors)

	target_indices: list[int] = [i for i, s in enumerate(result) if s == target]
	n_contract: int = round(fraction * len(target_indices))

	if n_contract == 0 or not target_indices:
		return result

	# Pool of candidate formal sectors excluding target and unemployed
	all_formal: list[str] = sorted(
		{s for s in result if s != unemployed_label and s != target}
	)
	if not all_formal:
		# Fallback: if no other formal sectors, redistribute to unemployed
		all_formal = [unemployed_label]

	chosen_indices: list[int] = r.sample(target_indices, n_contract)
	for idx in chosen_indices:
		result[idx] = r.choice(all_formal)

	return result


def build_uniform_sectors(labels: list[str], n_agents: int) -> list[str]:
	r"""
	Experiment C: Initialize agents uniformly across all possible states.

	Parameters
	----------
	labels : list[str]
		All sector labels including unemployed.
	n_agents : int
		Total simulated agent count.

	Returns
	-------
	list[str]
		Uniform list of sector assignments.
	"""
	if not labels or n_agents <= 0:
		return []

	clean_labels: list[str] = [str(lbl) for lbl in labels]
	n_labels: int = len(clean_labels)
	base_count: int = n_agents // n_labels
	remainder: int = n_agents % n_labels

	sectors: list[str] = []
	for i, label in enumerate(clean_labels):
		count: int = base_count + (1 if i < remainder else 0)
		sectors.extend([label] * count)

	return sectors


def resolve_target_sector(
	shares: pd.Series,
	target: str = "auto",
	unemployed_label: str = "unemployed",
) -> str:
	r"""
	Resolve target sector label for Experiment B.

	If ``target == "auto"``, returns the formal sector with maximum employment share.

	Parameters
	----------
	shares : pd.Series
		Sectoral shares.
	target : str
		Sector label or ``"auto"``.
	unemployed_label : str
		Label used for state s_0.

	Returns
	-------
	str
		Resolved sector label.
	"""
	if target != "auto":
		return str(target)

	formal_shares: pd.Series = shares[shares.index != unemployed_label]
	if formal_shares.empty:
		return unemployed_label
	return str(formal_shares.idxmax())
