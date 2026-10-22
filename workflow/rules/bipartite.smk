rule build_bipartite:
    """
    For a given time batch and granularity level, evaluate worker states
    at month t_end with primary job selection and save as Parquet.
    """
    input:
        marker=f"{config['paths']['data_parquet']}/.done",
        registry="data/processed/worker_registry.parquet",
    output:
        matrix="data/processed/{batch}/bipartite_g{gran}.parquet",
    params:
        start=lambda wc: int(wc.batch.split("_")[0]),
        end=lambda wc: int(wc.batch.split("_")[1]),
        gran=lambda wc: int(wc.gran),
        parquet_dir=config["paths"]["data_parquet"],
        max_working_age=config["pipeline"]["max_working_age"],
    log:
        "logs/bipartite/{batch}_g{gran}.log",
    script:
        "../../scripts/build_bipartite.py"
