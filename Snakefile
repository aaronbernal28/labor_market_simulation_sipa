"""
Snakefile — Labor Market Simulation SIPA
=========================================

Main orchestrator for the MLER transition matrix pipeline and ABM simulation.

Pipeline stages:
  1. preprocess_mler     — preprocess raw MLER CSV to partitioned Parquet & worker registry
  2. build_bipartite     — worker × sector occurrence matrices per period (YYYYMM)
  3. compute_transition  — NxN transition matrices between arbitrary (start, end) periods
  4. run_simulation      — ABM seeded by empirical matrices (scaffold)
  5. plot_results        — visualisations (scaffold)
"""

from snakemake.utils import validate

from src.pipeline.batch_utils import TimeBatchPartitioner

configfile: "config.yaml"
validate(config, schema="workflow/schemas/config.schema.yaml")

# ── Derived globals ────────────────────────────────────────────────────────────
_partitioner: TimeBatchPartitioner = TimeBatchPartitioner(
	start=config["pipeline"]["start_period"],
	end=config["pipeline"]["end_period"],
	interval=config["pipeline"]["time_interval"],
)

# Default batch intervals (e.g. ["199601_199612", "199701_199712", …])
BATCHES: list[str] = _partitioner.batch_ids()

# Unique 6-digit periods referenced by the default batches
PERIODS: list[int] = sorted(
	{p for b in _partitioner.batches() for p in (b.start, b.end)}
)

GRANULARITY: int = config["pipeline"]["granularity"]

# ── Wildcard constraints ───────────────────────────────────────────────────────
wildcard_constraints:
	period=r"\d{6}",
	start=r"\d{6}",
	end=r"\d{6}",
	gran=r"[1234]",

# ── Include modular rule files ─────────────────────────────────────────────────
include: "workflow/rules/extract.smk"
include: "workflow/rules/bipartite.smk"
include: "workflow/rules/transition.smk"
include: "workflow/rules/simulation.smk"
include: "workflow/rules/plots.smk"


# ── Default target ─────────────────────────────────────────────────────────────
rule all:
	input:
		# Preprocessing registry
		"data/processed/worker_registry.parquet",
		# Per-period bipartite worker states
		expand(
			"data/processed/{period}/bipartite_g{gran}.parquet",
			period=PERIODS,
			gran=GRANULARITY,
		),
		# Per-interval Hidalgo proximity matrices
		expand(
			"data/processed/{batch}/transition_matrix_g{gran}.parquet",
			batch=BATCHES,
			gran=GRANULARITY,
		),
		# Historical empirical series (Track 1)
		expand(
			"data/results/empirical_macro_series_g{gran}.csv",
			gran=GRANULARITY,
		),
		# Mesa ABM simulation trajectories (Track 2)
		expand(
			"data/results/sim_g{gran}/macro_trajectory.csv",
			gran=GRANULARITY,
		),
