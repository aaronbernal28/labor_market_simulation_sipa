"""
Script: build_bipartite.py
Called by: workflow/rules/bipartite.smk -> rule build_bipartite

Loads the partitioned MLER Parquet slice at the batch end month (t_end),
applies working-age filtering and primary job selection (max rem_tot),
and saves the resolved worker states to Parquet.
"""

import logging
import sys
from pathlib import Path

try:
	_sm = snakemake
except NameError:
	_sm = None

log_path = _sm.log[0] if _sm else "/dev/stderr"
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
	filename=log_path,
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.bipartite_builder import BipartiteBuilder
from src.pipeline.data_loader import MLERLoader


def main():
	parquet_dir = _sm.params.parquet_dir if _sm else sys.argv[1]
	out_path = _sm.output.matrix if _sm else sys.argv[2]
	start = int(_sm.params.start) if _sm else int(sys.argv[3])
	end = int(_sm.params.end) if _sm else int(sys.argv[4])
	gran = int(_sm.params.gran) if _sm else int(sys.argv[5])
	max_working_age = int(_sm.params.max_working_age) if _sm else 65

	log.info(
		"Loading MLER Parquet for batch [%d, %d] at observation month t_end=%d",
		start,
		end,
		end,
	)
	loader = MLERLoader(parquet_dir)
	df_end = loader.load_period(end)
	log.info("Loaded %d job relations at period %d", len(df_end), end)

	sector_series = loader.get_sector_column(gran, df_end)

	builder = BipartiteBuilder(
		df=df_end,
		granularity=gran,
		target_period=end,
		max_working_age=max_working_age,
	)
	states_df = builder.extract_worker_states(sector_series)
	log.info(
		"Extracted %d primary working-age worker states across %d sectors",
		len(states_df),
		states_df["sector"].nunique() if not states_df.empty else 0,
	)

	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	states_df.to_parquet(out_path, index=False)
	log.info("Saved batch worker states to %s", out_path)


if __name__ == "__main__":
	main()
