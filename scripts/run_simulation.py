"""
Script: run_simulation.py
Called by: workflow/rules/simulation.smk -> rule run_simulation

Runs the Mesa 3.5.1 Agent-Based Labor Market Model across the empirical
transition probability sequence for four experiments:
- baseline
- exp_a_layoff (unemployment mass shock)
- exp_b_contraction (sectoral contraction with uniform redistribution)
- exp_c_uniform (uniform initial distribution / maximum entropy)

Outputs strictly CSV trajectories for macro aggregates, sectoral shares,
and individual tracked agent paths.
"""

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.pipeline.empirical_series import compute_lilien_index
from src.simulation.labor_market import LaborMarketModel
from src.simulation.perturbations import (
	apply_mass_layoff,
	apply_sector_contraction,
	build_baseline_sectors,
	build_uniform_sectors,
	resolve_target_sector,
)

if TYPE_CHECKING:
	from snakemake.script import Snakemake

	snakemake: Snakemake

log_path: str = (
	snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else "/dev/stderr"
)
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log: logging.Logger = logging.getLogger(__name__)


def main() -> None:
	probs_paths: list[str] = [str(p) for p in snakemake.input.probs]
	sectoral_path: str = str(snakemake.input.sectoral)

	macro_path: str = str(snakemake.output.macro)
	sectoral_out_path: str = str(snakemake.output.sectoral)
	agents_path: str = str(snakemake.output.agents)

	batches: list[str] = [str(b) for b in snakemake.params.batches]
	sim_cfg: dict[str, Any] = snakemake.params.sim

	n_agents: int = int(sim_cfg["n_agents"])
	agent_sample_size: int = int(sim_cfg["agent_sample_size"])
	exp_a_shock_fraction: float = float(sim_cfg["exp_a_shock_fraction"])
	exp_b_target: str = str(sim_cfg["exp_b_target_sector"])
	exp_b_reduction_fraction: float = float(sim_cfg["exp_b_reduction_fraction"])
	seed: int = int(sim_cfg["seed"])

	log.info("Loading empirical sectoral series from %s", sectoral_path)
	emp_sectoral_df: pd.DataFrame = pd.read_csv(sectoral_path)

	start_period: int = int(batches[0].split("_")[0])
	t0_df: pd.DataFrame = emp_sectoral_df[emp_sectoral_df["period"] == start_period]
	init_shares: pd.Series = t0_df.set_index("sector")["share"]

	log.info(
		"Loading %d continuous batch probability matrices",
		len(batches),
	)
	batch_to_path: dict[str, str] = dict(zip(batches, probs_paths, strict=True))
	probs_sequence: list[tuple[int, pd.DataFrame]] = []
	all_labels_set: set[str] = set(init_shares.index)

	for b in batches:
		end_period: int = int(b.split("_")[1])
		p_path: str = batch_to_path[b]
		p_df: pd.DataFrame = pd.read_csv(p_path, index_col=0)
		probs_sequence.append((end_period, p_df))
		all_labels_set.update(str(s) for s in p_df.index)
		all_labels_set.update(str(s) for s in p_df.columns)

	all_sectors: list[str] = sorted(all_labels_set)

	# Build initial sector assignments for the 4 experiments
	log.info(
		"Initializing agent populations (n_agents=%d, seed=%d)",
		n_agents,
		seed,
	)
	sectors_baseline: list[str] = build_baseline_sectors(
		shares=init_shares,
		n_agents=n_agents,
	)

	rng_a = random.Random(seed + 101)
	sectors_exp_a: list[str] = apply_mass_layoff(
		sectors=sectors_baseline,
		fraction=exp_a_shock_fraction,
		rng=rng_a,
	)

	target_sector: str = resolve_target_sector(
		shares=init_shares,
		target=exp_b_target,
	)
	log.info("Experiment B target sector resolved to '%s'", target_sector)
	rng_b = random.Random(seed + 202)
	sectors_exp_b: list[str] = apply_sector_contraction(
		sectors=sectors_baseline,
		target=target_sector,
		fraction=exp_b_reduction_fraction,
		rng=rng_b,
	)

	sectors_exp_c: list[str] = build_uniform_sectors(
		labels=all_sectors,
		n_agents=n_agents,
	)

	experiments: list[tuple[str, list[str]]] = [
		("baseline", sectors_baseline),
		("exp_a_layoff", sectors_exp_a),
		("exp_b_contraction", sectors_exp_b),
		("exp_c_uniform", sectors_exp_c),
	]

	all_macro_dfs: list[pd.DataFrame] = []
	all_sectoral_records: list[dict[str, Any]] = []
	all_agent_dfs: list[pd.DataFrame] = []

	for exp_name, initial_sectors in experiments:
		log.info("Running simulation experiment: %s", exp_name)
		model: LaborMarketModel = LaborMarketModel(
			probs_sequence=probs_sequence,
			initial_sectors=initial_sectors,
			start_period=start_period,
			all_sectors=all_sectors,
			rng=seed,
		)

		for step_num in range(len(probs_sequence)):
			model.step()
			if not model.running:
				log.info(
					"Simulation stopped early at step %d",
					step_num + 1,
				)
				break

		# 1. Process macro trajectory
		macro_df: pd.DataFrame = model.datacollector.get_model_vars_dataframe()
		macro_df = macro_df.reset_index(drop=True)
		macro_df.insert(0, "experiment", exp_name)

		# Compute Lilien index across consecutive steps
		lilien_vals: list[float | None] = [None]
		for s in range(1, len(macro_df)):
			# Aggregate sectoral headcounts from sectoral history for steps s-1 and s
			h0 = pd.Series(
				{
					r["sector"]: r["headcount"]
					for r in model.sectoral_history
					if r["step"] == s - 1
				}
			)
			h1 = pd.Series(
				{
					r["sector"]: r["headcount"]
					for r in model.sectoral_history
					if r["step"] == s
				}
			)
			lilien_vals.append(compute_lilien_index(h0, h1))
		macro_df["lilien_index"] = lilien_vals
		all_macro_dfs.append(macro_df)

		# 2. Process sectoral trajectory
		for r in model.sectoral_history:
			rec = dict(r)
			rec["experiment"] = exp_name
			all_sectoral_records.append(rec)

		# 3. Process agent sample trajectory
		agent_vars_df: pd.DataFrame = (
			model.datacollector.get_agent_vars_dataframe().reset_index()
		)
		# Step column comes as Step (float or int)
		step_col: str = "Step" if "Step" in agent_vars_df.columns else "step"
		id_col: str = "AgentID" if "AgentID" in agent_vars_df.columns else "agent_id"

		unique_ids: list[Any] = list(agent_vars_df[id_col].unique())
		sample_ids: set[Any] = set(unique_ids[:agent_sample_size])

		sample_agent_df: pd.DataFrame = agent_vars_df[
			agent_vars_df[id_col].isin(sample_ids)
		].copy()
		sample_agent_df.insert(0, "experiment", exp_name)
		sample_agent_df = sample_agent_df.rename(
			columns={step_col: "step", id_col: "agent_id"},
		)
		sample_agent_df["step"] = sample_agent_df["step"].astype(int)

		# Map step to period
		step_to_period: dict[int, int] = dict(
			zip(macro_df["step"], macro_df["period"], strict=True)
		)
		sample_agent_df["period"] = sample_agent_df["step"].map(step_to_period)

		# Reorder columns
		sample_agent_df = sample_agent_df[
			["experiment", "step", "period", "agent_id", "sector", "tenure"]
		]
		all_agent_dfs.append(sample_agent_df)

	# Combine all experiment runs
	final_macro_df: pd.DataFrame = pd.concat(all_macro_dfs, ignore_index=True)
	final_sectoral_df: pd.DataFrame = pd.DataFrame(all_sectoral_records)
	# Reorder sectoral columns
	final_sectoral_df = final_sectoral_df[
		["experiment", "step", "period", "sector", "headcount", "share"]
	]
	final_agents_df: pd.DataFrame = pd.concat(all_agent_dfs, ignore_index=True)

	Path(macro_path).parent.mkdir(parents=True, exist_ok=True)
	Path(sectoral_out_path).parent.mkdir(parents=True, exist_ok=True)
	Path(agents_path).parent.mkdir(parents=True, exist_ok=True)

	final_macro_df.to_csv(macro_path, index=False)
	final_sectoral_df.to_csv(sectoral_out_path, index=False)
	final_agents_df.to_csv(agents_path, index=False)

	log.info("Saved macro trajectory to %s", macro_path)
	log.info("Saved sectoral trajectory to %s", sectoral_out_path)
	log.info("Saved agent sample trajectory to %s", agents_path)


if __name__ == "__main__":
	main()
