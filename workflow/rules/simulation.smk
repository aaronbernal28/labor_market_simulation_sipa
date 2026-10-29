rule compute_empirical_series:
	"""
	Compute historical empirical state representation S_t, Shannon entropy H_t,
	net sectoral variation Delta x_t, and Lilien reallocation index L_t
	across all observed periods in SIPA (Track 1).
	"""
	input:
		states=expand("data/processed/{period}/bipartite_g{{gran}}.parquet", period=PERIODS),
		registry="data/processed/worker_registry.parquet",
	output:
		macro="data/results/empirical_macro_series_g{gran}.csv",
		sectoral="data/results/empirical_sectoral_series_g{gran}.csv",
	log:
		"logs/empirical_series_g{gran}.log",
	params:
		periods=PERIODS,
		max_working_age=config["pipeline"]["max_working_age"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/compute_empirical_series.py"


rule run_simulation:
	"""
	Run the Mesa 3.5.1 Agent-Based Labor Market Model across the empirical
	continuous transition probability sequence for baseline and perturbation experiments.
	"""
	input:
		probs=expand("data/processed/{batch}/transition_probs_g{{gran}}.csv", batch=BATCHES),
		sectoral="data/results/empirical_sectoral_series_g{gran}.csv",
	output:
		macro="data/results/sim_g{gran}/macro_trajectory.csv",
		sectoral="data/results/sim_g{gran}/sectoral_trajectory.csv",
		agents="data/results/sim_g{gran}/agent_sample_trajectory.csv",
	log:
		"logs/simulation_g{gran}.log",
	params:
		batches=BATCHES,
		sim=config["simulation"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/run_simulation.py"
