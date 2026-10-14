"""Step 0 reference analysis: recomputes every expected value in `answers.yaml` with plain pandas.

EVALUATOR-ONLY, like the rest of this directory: never mounted into a container, never
put in a prompt. Run it from the eda-lite folder to check the answer key against the data:

    uv run python projects/stix_flares_simple/evaluation_spec/reference_analysis.py

It prints one line per claim and exits non-zero if any value disagrees with `answers.yaml` (same
matching rule as the grader).
Each claim is one expression on one column, as defined in `project.yaml`.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from eda_lite.grade import matches

HERE = Path(__file__).parent


def reference_values(frame: pd.DataFrame) -> dict:
    mean_class = frame["goes_estimated_mean_class"]
    return {
        "n_rows": len(frame),
        "n_columns": frame.shape[1],
        "n_duplicate_flare_ids": int(frame["flare_id"].duplicated().sum()),
        "max_counts_4_10_keV": int(frame["4-10 keV"].max()),
        "median_counts_4_10_keV": float(frame["4-10 keV"].median()),
        "most_common_estimated_class": mean_class.value_counts().idxmax(),
        "n_strong_flares": int(mean_class.str[0].isin(["M", "X"]).sum()),
        "n_attenuator_flares": int(frame["att_in"].sum()),
        "n_imaging_errors": int(frame["error_with_imaging"].sum()),
        "mean_solo_distance_au": float(frame["solo_position_AU_distance"].mean()),
    }


def main() -> int:
    frame = pd.read_csv(HERE.parent / "data" / "stix_flares.csv")
    computed = reference_values(frame)
    answers = yaml.safe_load((HERE / "answers.yaml").read_text())
    disagreements = 0
    for claim_id, expected in answers.items():
        actual = computed.get(claim_id)
        agrees = actual is not None and matches(expected, actual)
        disagreements += not agrees
        print(f"{'ok ' if agrees else 'DIFF'} {claim_id:<28} {expected!s:<10} {actual}")
    unchecked = sorted(set(computed) - set(answers))
    if unchecked:
        print(f"computed but not in answers.yaml: {unchecked}")
    return int(bool(disagreements))


if __name__ == "__main__":
    sys.exit(main())
