"""
Script: plot_matrices.py
Called by: workflow/rules/plots.smk → rule plot_results

Generates heatmap visualisations of transition matrices.
Full implementation TBD.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

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
	series_path: str = str(snakemake.input.series)
	out_dir: str = str(snakemake.output.figures)

	log.info("Loading transition series from %s", series_path)
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
