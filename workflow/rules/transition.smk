def _next_batch_bipartite(wildcards):
    """
    Resolve the bipartite file for the next time batch (t+1),
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
    Compute the empirical (N+1) x (N+1) Hidalgo proximity / transition
    matrix between consecutive batches t and t+1, including the
    unemployment state s_0.
    """
    input:
        current="data/processed/{batch}/bipartite_g{gran}.parquet",
        next_b=_next_batch_bipartite,
        registry="data/processed/worker_registry.parquet",
    output:
        matrix="data/processed/{batch}/transition_matrix_g{gran}.parquet",
    params:
        start=lambda wc: int(wc.batch.split("_")[0]),
        end=lambda wc: int(wc.batch.split("_")[1]),
        gran=lambda wc: int(wc.gran),
        metric=config["pipeline"]["transition_metric"],
        max_working_age=config["pipeline"]["max_working_age"],
    log:
        "logs/transition/{batch}_g{gran}.log",
    script:
        "../../scripts/compute_transition.py"
