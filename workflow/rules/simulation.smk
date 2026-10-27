rule run_simulation:
	"""
	Run the ABM labor market simulation seeded with the empirical
	transition series.

	Parameters come from config['simulation'].
	Full implementation TBD — currently produces placeholder output.
	"""
	input:
		matrices="data/results/transition_series_g{gran}.parquet",
	output:
		results="data/results/sim_g{gran}/simulation_output.parquet",
		flag=touch("data/results/sim_g{gran}/.done"),
	log:
		"logs/simulation_g{gran}.log",
	params:
		rate_fired=config["simulation"]["rate_fired"],
		rate_vacancies=config["simulation"]["rate_vacancies"],
		initial_employed_fraction=config["simulation"]["initial_employed_fraction"],
		n_steps=config["simulation"]["n_steps"],
		seed=config["simulation"]["seed"],
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/run_simulation.py"
