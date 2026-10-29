r"""
Labor Market Agent-Based Model (Mesa 3.5.1).

Simulates individual worker agent transitions across continuous time batches
guided by precomputed empirical transition probability matrices.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import mesa
import numpy as np
import pandas as pd

from src.pipeline.empirical_series import (
	compute_shannon_entropy,
	compute_shares,
)

if TYPE_CHECKING:
	import numpy.typing as npt


class WorkerAgent(mesa.Agent):
	r"""
	Individual worker agent in the labor market.

	Parameters
	----------
	model : LaborMarketModel
		Parent simulation model instance.
	sector : str
		Initial sector assignment (e.g. ``"unemployed"``, ``"A"``, ``"52"``).
	"""

	def __init__(self, model: LaborMarketModel, sector: str) -> None:
		super().__init__(model)
		self.sector: str = str(sector)
		self.tenure: int = 0

	def step(self) -> None:
		r"""
		Transition agent to next sector according to model's current probability matrix.

		Notes
		-----
		# FIXME: If the worker's current sector is not present in the current batch
		# transition matrix, the worker remains in place and increments tenure.
		"""
		probs_df: pd.DataFrame = self.model.current_probs
		if probs_df.empty or self.sector not in probs_df.index:
			self.tenure += 1
			return

		row: pd.Series = probs_df.loc[self.sector]
		pop: list[str] = [str(col) for col in row.index]
		weights: npt.NDArray[np.float64] = row.to_numpy(dtype=np.float64)

		# Sample next sector using model's seeded RNG
		new_sector: str = self.model.random.choices(pop, weights=weights, k=1)[0]

		if new_sector == self.sector:
			self.tenure += 1
		else:
			self.sector = new_sector
			self.tenure = 0


class LaborMarketModel(mesa.Model):
	r"""
	Agent-based labor market simulation model.

	Parameters
	----------
	probs_sequence : list[tuple[int, pd.DataFrame]]
		Ordered list of ``(period_end, transition_probs_df)`` for each continuous batch.
	initial_sectors : list[str]
		Sector assignment for each worker agent at step 0.
	start_period : int
		Period identifier for step 0 (e.g. 199601).
	all_sectors : list[str] | None
		Universal list of sector codes. Inferred if None.
	unemployed_label : str
		Label used for state s_0 (default ``"unemployed"``).
	rng : int
		Random seed for reproducible transitions.
	"""

	def __init__(
		self,
		probs_sequence: list[tuple[int, pd.DataFrame]],
		initial_sectors: list[str],
		start_period: int,
		all_sectors: list[str] | None = None,
		unemployed_label: str = "unemployed",
		rng: int = 42,
	) -> None:
		super().__init__(rng=rng)
		self.probs_sequence: list[tuple[int, pd.DataFrame]] = probs_sequence
		self.start_period: int = start_period
		self.period: int = start_period
		self.unemployed_label: str = unemployed_label
		self.current_probs: pd.DataFrame = (
			probs_sequence[0][1] if probs_sequence else pd.DataFrame()
		)

		if all_sectors is not None:
			self.all_sectors: list[str] = sorted(all_sectors)
		else:
			seen_sectors: set[str] = set(initial_sectors)
			for _, p in probs_sequence:
				seen_sectors.update(str(s) for s in p.index)
				seen_sectors.update(str(s) for s in p.columns)
			self.all_sectors = sorted(seen_sectors)

		# Create worker agents
		for sector in initial_sectors:
			WorkerAgent(self, sector=sector)

		self._update_headcounts()

		# Setup DataCollector
		self.datacollector: mesa.DataCollector = mesa.DataCollector(
			model_reporters={
				"period": lambda m: m.period,
				"step": lambda m: m.steps,
				"active_pool": lambda m: len(m.agents),
				"formal_employed": lambda m: m.current_formal_employed,
				"unemployed": lambda m: m.current_unemployed,
				"unemployment_rate": lambda m: m.current_unemployment_rate,
				"entropy": lambda m: m.current_entropy,
			},
			agent_reporters={
				"sector": "sector",
				"tenure": "tenure",
			},
		)

		# Initial recording at step 0
		self.datacollector.collect(self)
		self.sectoral_history: list[dict[str, int | str | float]] = []
		self._record_sectoral_history()

	def _update_headcounts(self) -> None:
		"""Update summary headcounts and macro metrics for fast reporter access."""
		counts_dict: Counter[str] = Counter(a.sector for a in self.agents)
		self.current_headcounts: pd.Series = pd.Series(
			{s: counts_dict.get(s, 0) for s in self.all_sectors}, dtype=np.float64
		)
		self.current_shares: pd.Series = compute_shares(self.current_headcounts)

		self.current_unemployed: int = counts_dict.get(self.unemployed_label, 0)
		total_agents: int = len(self.agents)
		self.current_formal_employed: int = total_agents - self.current_unemployed
		self.current_unemployment_rate: float = (
			self.current_unemployed / total_agents if total_agents > 0 else 0.0
		)
		self.current_entropy: float = compute_shannon_entropy(self.current_shares)

	def _record_sectoral_history(self) -> None:
		"""Record current sectoral headcounts and shares for export."""
		for sec in self.all_sectors:
			count: float = float(self.current_headcounts.get(sec, 0.0))
			share: float = float(self.current_shares.get(sec, 0.0))
			self.sectoral_history.append(
				{
					"step": self.steps,
					"period": self.period,
					"sector": sec,
					"headcount": count,
					"share": share,
				}
			)

	def step(self) -> None:
		r"""Advance simulation by one batch."""
		if self.steps > len(self.probs_sequence):
			self.running = False
			return

		# Select probability matrix for current step (batch index self.steps - 1)
		idx: int = self.steps - 1
		if idx >= len(self.probs_sequence):
			self.running = False
			return

		self.period, self.current_probs = self.probs_sequence[idx]

		# Shuffle and execute agent transitions
		self.agents.shuffle_do("step")

		# Update macro states and collect data
		self._update_headcounts()
		self.datacollector.collect(self)
		self._record_sectoral_history()

		if self.steps >= len(self.probs_sequence):
			self.running = False
