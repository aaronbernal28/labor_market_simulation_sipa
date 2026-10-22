rule preprocess_mler:
    """
    Preprocess raw MLER CSV into year-partitioned Parquet files
    and build the working-age worker registry.
    """
    input:
        csv=config["datasets"]["sipa"]["source"],
    output:
        marker=touch(f"{config['paths']['data_parquet']}/.done"),
        registry="data/processed/worker_registry.parquet",
    params:
        out_dir=config["paths"]["data_parquet"],
        max_working_age=config["pipeline"]["max_working_age"],
    log:
        "logs/preprocess_mler.log",
    script:
        "../../scripts/preprocess_data.py"
