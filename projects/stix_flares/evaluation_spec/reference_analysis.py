"""Step 0 reference analysis: recomputes every expected value in `answers.yaml` with plain pandas.

EVALUATOR-ONLY, like the rest of this directory: never mounted into a container, never
put in a prompt. Run it from the eda-lite folder to check the answer key against the data:

    uv run python projects/stix_flares/evaluation_spec/reference_analysis.py

It prints one line per claim and exits non-zero if any value disagrees with `answers.yaml` (same
matching rule as the grader).
The definitions follow `project.yaml`: a flare's month and year come from `peak_UTC`, and a
flare is strong when `goes_estimated_mean_class` starts with M or X.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from eda_lite.grade import matches

HERE = Path(__file__).parent
YEARS = range(2021, 2026)


def reference_values(frame: pd.DataFrame) -> dict:
    peak = pd.to_datetime(frame["peak_UTC"])
    month = peak.dt.strftime("%Y-%m")
    year = peak.dt.year
    strong = frame["goes_estimated_mean_class"].astype("string").str[0].isin(["M", "X"])
    per_month = month.value_counts()

    values = {
        "n_flares": len(frame),
        "n_duplicate_flare_ids": int(frame["flare_id"].duplicated().sum()),
        "n_months": int(per_month.size),
        "busiest_month": per_month.idxmax(),
        "busiest_month_count": int(per_month.max()),
        "n_strong_flares": int(strong.sum()),
        "strong_share_overall": float(strong.mean()),
    }
    for y in YEARS:
        values[f"flares_{y}"] = int((year == y).sum())
        values[f"strong_share_{y}"] = float(strong[year == y].mean())
    return values


def main() -> int:
    frame = pd.read_csv(HERE.parent / "data" / "stix_flares.csv")
    computed = reference_values(frame)
    answers = yaml.safe_load((HERE / "answers.yaml").read_text())
    disagreements = 0
    for claim_id, expected in answers.items():
        actual = computed.get(claim_id)
        agrees = actual is not None and matches(expected, actual)
        disagreements += not agrees
        print(f"{'ok ' if agrees else 'DIFF'} {claim_id:<24} {expected!s:<10} {actual}")
    unchecked = sorted(set(computed) - set(answers))
    if unchecked:
        print(f"computed but not in answers.yaml: {unchecked}")
    return int(bool(disagreements))


if __name__ == "__main__":
    sys.exit(main())
