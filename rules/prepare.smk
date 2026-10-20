"""Prepare data for the labor market simulation."""

rule unzip:
	input:
		"data/mler_csv.rar"
	output:
		"data/mler_csv"
	shell:
		"unrar x {input} {output}"

rule clean_data:
	# TODO: Add a rule to clean the data and save it to data/processed/sipa_cleaned.csv
	input:
		"data/mler_csv"
	output:
		"data/processed/sipa_cleaned.csv"
	shell:
		"python scripts/clean_sipa.py"
