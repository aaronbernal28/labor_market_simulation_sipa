rule build_bipartite:
	"""
	For a given period (YYYYMM) and granularity level, evaluate worker states
	with working-age filtering and primary job selection and save as Parquet.
	"""
	input:
		marker=f"{config['paths']['data_parquet']}/.done",
		registry="data/processed/worker_registry.parquet",
	output:
		matrix="data/processed/{period}/bipartite_g{gran}.parquet",
	log:
		"logs/bipartite/{period}_g{gran}.log",
	params:
		period=lambda wc: int(wc.period),
		gran=lambda wc: int(wc.gran),
		parquet_dir=lambda wc, input: str(Path(input.marker).parent),
		max_working_age=config["pipeline"]["max_working_age"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/build_bipartite.py"
