"""
Script: plot_matrices.py
Called by: workflow/rules/plots.smk → rule plot_results

Generates heatmap visualisations of transition matrices.
Full implementation TBD.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
	_sm = snakemake
except NameError:
	_sm = None

log_path = _sm.log[0] if _sm else "/dev/stderr"
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
	series_path = _sm.input.series if _sm else sys.argv[1]
	out_dir = _sm.output.figures if _sm else sys.argv[2]

	log.info("Loading transition series from %s", series_path)
	# TODO: load and iterate over batch-specific matrices

	Path(out_dir).mkdir(parents=True, exist_ok=True)

	# Placeholder: save a single summary figure
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.set_title("Transition Matrix Heatmap — placeholder")
	ax.text(0.5, 0.5, "TBD", ha="center", va="center", fontsize=24, color="grey")
	fig.savefig(Path(out_dir) / "placeholder.png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	log.info("Saved placeholder figure to %s", out_dir)


if __name__ == "__main__":
	main()
