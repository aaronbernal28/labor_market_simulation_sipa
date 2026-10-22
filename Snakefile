"""
Snakefile — Labor Market Simulation SIPA
=========================================

Main orchestrator for the MLER transition matrix pipeline and ABM simulation.

Pipeline stages:
  1. extract_raw     — unpack MLER RAR to data/raw/
  2. build_bipartite — worker × sector occurrence matrices per time batch
  3. compute_transition — NxN transition matrices between consecutive batches
  4. aggregate_matrices — collect all batch matrices into one series
  5. run_simulation  — ABM seeded by empirical matrices (scaffold)
  6. plot_results    — visualisations (scaffold)
"""

import sys
from pathlib import Path

configfile: "config.yaml"

# ── Make project root importable ───────────────────────────────────────────────
sys.path.insert(0, str(Path(workflow.basedir)))

from src.pipeline.batch_utils import TimeBatchPartitioner

# ── Derived globals ────────────────────────────────────────────────────────────
_partitioner = TimeBatchPartitioner(
    start=config["pipeline"]["start_period"],
    end=config["pipeline"]["end_period"],
    interval=config["pipeline"]["time_interval"],
)

# All batch IDs (e.g. ["199601_199612", "199701_199712", …])
BATCHES = _partitioner.batch_ids()

# Batches for which a transition can be computed (all except the last,
# because computing T for batch t requires bipartite data from batch t+1).
TRANSITION_BATCHES = BATCHES[:-1]

GRANULARITY = config["pipeline"]["granularity"]

# ── Wildcard constraints ───────────────────────────────────────────────────────
wildcard_constraints:
    batch=r"\d{6}_\d{6}",
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
        # Per-batch bipartite worker states
        expand(
            "data/processed/{batch}/bipartite_g{gran}.parquet",
            batch=BATCHES,
            gran=GRANULARITY,
        ),
        # Per-batch Hidalgo proximity matrices (all except last batch)
        expand(
            "data/processed/{batch}/transition_matrix_g{gran}.parquet",
            batch=TRANSITION_BATCHES,
            gran=GRANULARITY,
        ),
