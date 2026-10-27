"""
Labor market ABM seeded by empirical transition matrices.

Full implementation TBD — parameters below are scaffolded.  The model
reads a time-indexed series of sector-to-sector transition matrices
produced by the pipeline and uses them to drive agent transitions.
"""

from __future__ import annotations

import mesa
import pandas as pd


class LaborMarketSimulation(mesa.Model):
	"""
	ABM labor market model.

	Parameters
	----------
	transition_series : dict[str, pd.DataFrame]
	    Mapping ``batch_id`` → NxN stochastic transition matrix.
	rate_fired : float
	    Monthly probability of involuntary job loss.
	rate_vacancies : float
	    Monthly rate at which new vacancies open per sector.
	initial_employed_fraction : float
	    Share of workers initially employed.
	n_steps : int
	    Simulation horizon in months.
	seed : int
	    Random seed for reproducibility.

	# ── TBD parameters (not yet defined) ───────────────────────────────
	# matching_function : str     e.g. 'random', 'closest_sector'
	# wage_bargaining  : str     e.g. 'nash', 'posted_wage'
	# search_radius    : int     sectors a worker can reach in one step
	"""

	def __init__(
		self,
		transition_series: dict[str, pd.DataFrame] | None = None,
		rate_fired: float = 0.02,
		rate_vacancies: float = 0.03,
		initial_employed_fraction: float = 0.85,
		n_steps: int = 120,
		seed: int = 42,
	) -> None:
		super().__init__(seed=seed)
		self.transition_series: dict[str, pd.DataFrame] = transition_series or {}
		self.rate_fired: float = rate_fired
		self.rate_vacancies: float = rate_vacancies
		self.initial_employed_fraction: float = initial_employed_fraction
		self.n_steps: int = n_steps
		self.running: bool = True
		self.current_step: int = 0

	def step(self) -> None:
		"""Single simulation tick. TODO: implement full ABM logic."""
		self.current_step += 1
		# TODO: apply transitions, firings, vacancy matching
		if self.current_step >= self.n_steps:
			self.running = False
