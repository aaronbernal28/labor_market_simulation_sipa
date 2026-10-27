"""
Script: analyze_nulls.py
========================
Analyzes missing/null values and distributions in data/MLER.csv,
with a focus on fnacim (birth year) and worker demographics.
"""

import csv
import sys
from collections import defaultdict


def analyze_mler_csv(csv_path: str, max_rows: int | None = 500_000) -> None:
	print(f"Scanning {csv_path} (limit={max_rows} rows)...")
	missing_counts: defaultdict[str, int] = defaultdict(int)
	total_rows: int = 0
	birth_years: list[int] = []
	weights: list[float] = []

	with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
		reader: csv.DictReader = csv.DictReader(f)
		for row in reader:
			total_rows += 1
			for k, v in row.items():
				if v is None or v.strip() == "":
					missing_counts[k] += 1

			fn: str = row.get("fnacim", "").strip()
			if fn:
				try:
					birth_years.append(int(fn))
				except ValueError:
					pass

			p: str = row.get("pondera", "").strip()
			if p:
				try:
					weights.append(float(p))
				except ValueError:
					pass

			if max_rows and total_rows >= max_rows:
				break

	print(f"\n--- Analysis Summary ({total_rows:,} rows scanned) ---")
	print("Column Missingness:")
	if reader.fieldnames:
		for col in reader.fieldnames:
			cnt: int = missing_counts[col]
			pct: float = (cnt / total_rows) * 100
			print(f"  - {col:15s}: {cnt:8,d} missing ({pct:5.2f}%)")

	if birth_years:
		print("\nBirth Year (fnacim) Distribution:")
		print(f"  - Min birth year : {min(birth_years)}")
		print(f"  - Max birth year : {max(birth_years)}")
		p25: int = sorted(birth_years)[len(birth_years) // 4]
		p50: int = sorted(birth_years)[len(birth_years) // 2]
		p75: int = sorted(birth_years)[(3 * len(birth_years)) // 4]
		print(f"  - Quartiles      : P25={p25}, Median={p50}, P75={p75}")

	if weights:
		print("\nExpansion Weights (pondera):")
		print(f"  - Min pondera    : {min(weights):.4f}")
		print(f"  - Max pondera    : {max(weights):.4f}")
		print(f"  - Distinct values: {len(set(weights))}")


if __name__ == "__main__":
	csv_file: str = sys.argv[1] if len(sys.argv) > 1 else "data/MLER.csv"
	analyze_mler_csv(csv_file, max_rows=500_000)
