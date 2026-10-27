"""
Pipeline modules for MLER data processing and transition matrix generation.

Only ``TimeBatch`` and ``TimeBatchPartitioner`` are imported eagerly
(pure Python, no external dependencies).  Heavy classes that depend on
pandas/scipy/numpy are available via lazy imports or by importing
the submodules directly.
"""

from src.pipeline.batch_utils import TimeBatch, TimeBatchPartitioner

__all__ = [
	"TimeBatch",
	"TimeBatchPartitioner",
	# Available via direct submodule import:
	# from src.pipeline.data_loader import MLERLoader
	# from src.pipeline.bipartite_builder import BipartiteBuilder
	# from src.pipeline.transition_builder import BipartiteToTransition
]
