def next_batch_bipartite(wildcards):
	"""
	Resolve the bipartite file for the next time batch (t+1),
	required to compute the transition matrix for batch t.
	"""
	current_id = wildcards.batch
	gran = wildcards.gran
	try:
		idx = BATCHES.index(current_id)
	except ValueError:
		raise ValueError(
			f"Batch '{current_id}' not found in BATCHES list. "
			f"Available: {BATCHES}"
		)
	if idx + 1 >= len(BATCHES):
		raise ValueError(
			f"Batch '{current_id}' is the last batch — no t+1 available "
			f"for transition computation."
		)
	next_id = BATCHES[idx + 1]
	return f"data/processed/{next_id}/bipartite_g{gran}.parquet"
