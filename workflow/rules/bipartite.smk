rule build_bipartite:
    """
    For a given time batch and granularity level, build the bipartite
    worker × sector occurrence matrix and save it as Parquet.
    """
    input:
        csv="data/raw/mler.csv",
    output:
        matrix="data/processed/{batch}/bipartite_g{gran}.parquet",
    params:
        start=lambda wc: wc.batch.split("_")[0],
        end=lambda wc: wc.batch.split("_")[1],
        gran=lambda wc: int(wc.gran),
    log:
        "logs/bipartite/{batch}_g{gran}.log",
    script:
        "../../scripts/build_bipartite.py"
