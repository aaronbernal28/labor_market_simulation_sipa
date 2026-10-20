def _next_batch_bipartite(wildcards):
    """
    Resolve the bipartite file for the *next* time batch (t+1),
    required to compute the transition matrix for batch t.
    """
    current_id = wildcards.batch
    gran = wildcards.gran
    try:
        idx = BATCHES.index(current_id)
    except ValueError:
        raise ValueError(
            f"Batch '{current_id}' not found in BATCHES list. "
            f"Available: {BATCHES}"
        )
    if idx + 1 >= len(BATCHES):
        raise ValueError(
            f"Batch '{current_id}' is the last batch — no t+1 available "
            f"for transition computation."
        )
    next_id = BATCHES[idx + 1]
    return f"data/processed/{next_id}/bipartite_g{gran}.parquet"


rule compute_transition:
    """
    Convert a bipartite occurrence matrix into an NxN sector transition
    matrix.  Requires both the current batch and the next batch bipartite
    files.

    The core logic is delegated to BipartiteToTransition
    (src/pipeline/transition_builder.py).  Its compute() method is
    currently a PLACEHOLDER — user will supply real formula.
    """
    input:
        current="data/processed/{batch}/bipartite_g{gran}.parquet",
        next_b=_next_batch_bipartite,
    output:
        matrix="data/processed/{batch}/transition_matrix_g{gran}.parquet",
    log:
        "logs/transition/{batch}_g{gran}.log",
    script:
        "../../scripts/compute_transition.py"


rule aggregate_matrices:
    """
    Collect all per-batch transition matrices into a single
    time-indexed Parquet series.
    """
    input:
        matrices=expand(
            "data/processed/{batch}/transition_matrix_g{{gran}}.parquet",
            batch=TRANSITION_BATCHES,
        ),
    output:
        series="data/results/transition_series_g{gran}.parquet",
    log:
        "logs/aggregate_g{gran}.log",
    script:
        "../../scripts/aggregate_matrices.py"
