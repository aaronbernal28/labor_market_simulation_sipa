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
	log:
		"logs/bipartite/{batch}_g{gran}.log",
	params:
		start=lambda wc: int(wc.batch.split("_")[0]),
		end=lambda wc: int(wc.batch.split("_")[1]),
		gran=lambda wc: int(wc.gran),
		parquet_dir=lambda wc, input: str(Path(input.marker).parent),
		max_working_age=config["pipeline"]["max_working_age"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/build_bipartite.py"
