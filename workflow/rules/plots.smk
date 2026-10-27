rule plot_results:
	"""
	Generate visualisations of transition matrices and simulation results.
	Full implementation TBD.
	"""
	input:
		series="data/results/transition_series_g{gran}.parquet",
	output:
		figures=directory("data/graphs/transition_g{gran}"),
	log:
		"logs/plots_g{gran}.log",
	conda:
		"../envs/default.yaml"
	script:
		"../../scripts/plot_matrices.py"
