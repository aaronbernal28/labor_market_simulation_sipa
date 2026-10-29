"""Simulation modules for the labor market ABM."""

from src.simulation.labor_market import LaborMarketModel, WorkerAgent
from src.simulation.perturbations import (
	apply_mass_layoff,
	apply_sector_contraction,
	build_baseline_sectors,
	build_uniform_sectors,
	resolve_target_sector,
)

# Backwards compatibility alias
LaborMarketSimulation = LaborMarketModel

__all__ = [
	"LaborMarketModel",
	"LaborMarketSimulation",
	"WorkerAgent",
	"apply_mass_layoff",
	"apply_sector_contraction",
	"build_baseline_sectors",
	"build_uniform_sectors",
	"resolve_target_sector",
]
