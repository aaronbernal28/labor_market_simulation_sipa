rule compute_transition:
	"""
	Compute the empirical (N+1) x (N+1) Hidalgo proximity / transition
	matrix between arbitrary periods start (t) and end (t+1), including the
	unemployment state s_0.
	"""
	input:
		current="data/processed/{start}/bipartite_g{gran}.parquet",
		next_b="data/processed/{end}/bipartite_g{gran}.parquet",
		registry="data/processed/worker_registry.parquet",
	output:
		matrix="data/processed/{start}_{end}/transition_matrix_g{gran}.parquet",
	log:
		"logs/transition/{start}_{end}_g{gran}.log",
	params:
		start=lambda wc: int(wc.start),
		end=lambda wc: int(wc.end),
		gran=lambda wc: int(wc.gran),
		metric=config["pipeline"]["transition_metric"],
		max_working_age=config["pipeline"]["max_working_age"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/compute_transition.py"
