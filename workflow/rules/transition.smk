rule compute_transition:
	"""
	Compute the empirical (N+1) x (N+1) Hidalgo proximity / transition
	matrix between consecutive batches t and t+1, including the
	unemployment state s_0.
	"""
	input:
		current="data/processed/{batch}/bipartite_g{gran}.parquet",
		next_b=next_batch_bipartite,
		registry="data/processed/worker_registry.parquet",
	output:
		matrix="data/processed/{batch}/transition_matrix_g{gran}.parquet",
	log:
		"logs/transition/{batch}_g{gran}.log",
	params:
		start=lambda wc: int(wc.batch.split("_")[0]),
		end=lambda wc: int(wc.batch.split("_")[1]),
		gran=lambda wc: int(wc.gran),
		metric=config["pipeline"]["transition_metric"],
		max_working_age=config["pipeline"]["max_working_age"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/compute_transition.py"
