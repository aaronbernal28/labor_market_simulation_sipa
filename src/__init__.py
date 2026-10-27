"""
Labor Market Simulation SIPA - Core Domain Library.

Tier 1 pure domain package for MLER microdata processing,
bipartite state construction, and Hidalgo proximity transitions.
"""

from typing import Any

from src.pipeline.batch_utils import TimeBatch, TimeBatchPartitioner

__all__ = [
	"BipartiteBuilder",
	"BipartiteToTransition",
	"LaborMarketSimulation",
	"MLERLoader",
	"TimeBatch",
	"TimeBatchPartitioner",
]


def __getattr__(name: str) -> Any:
	if name == "MLERLoader":
		from src.pipeline.data_loader import MLERLoader

		return MLERLoader
	if name == "BipartiteBuilder":
		from src.pipeline.bipartite_builder import BipartiteBuilder

		return BipartiteBuilder
	if name == "BipartiteToTransition":
		from src.pipeline.transition_builder import BipartiteToTransition

		return BipartiteToTransition
	if name == "LaborMarketSimulation":
		from src.simulation.labor_market import LaborMarketSimulation

		return LaborMarketSimulation
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
