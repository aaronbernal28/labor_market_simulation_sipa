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


rule export_transition_csv:
	"""
	Export a computed Parquet transition matrix to human-readable CSV with
	official sector labels mapped from data/anexo_estadistico_y_descriptores.xlsx.
	"""
	input:
		matrix="data/processed/{start}_{end}/transition_matrix_g{gran}.parquet",
		descriptors="data/anexo_estadistico_y_descriptores.xlsx",
	output:
		csv="data/processed/{start}_{end}/transition_matrix_g{gran}.csv",
	log:
		"logs/transition/{start}_{end}_g{gran}_csv.log",
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/export_transition_csv.py"


rule normalize_transition:
	"""
	Row-normalize a computed Hidalgo proximity matrix into fixed transition
	probabilities (CSV) for agent random sampling in the Mesa ABM.
	"""
	input:
		matrix="data/processed/{start}_{end}/transition_matrix_g{gran}.parquet",
	output:
		probs="data/processed/{start}_{end}/transition_probs_g{gran}.csv",
	log:
		"logs/transition/{start}_{end}_g{gran}_probs.log",
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/normalize_transition.py"
