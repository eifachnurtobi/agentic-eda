import sys
from pathlib import Path

import pandas as pd
import yaml

from eda_lite.grade import matches

HERE = Path(__file__).parent
CLASS_BASE_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
CPD_WINDOW = r"cpd_(\d{8}T\d{6})-(\d{8}T\d{6})_"


def class_to_flux(classes: pd.Series) -> pd.Series:

    parts = classes.astype("string").str.extract(r"^([ABCMX])(\d+(?:\.\d+)?)$")
    return parts[0].map(CLASS_BASE_FLUX).astype(float) * parts[1].astype(float)


def reference_values(frame: pd.DataFrame) -> dict:
    start = pd.to_datetime(frame["start_UTC"])
    end = pd.to_datetime(frame["end_UTC"])
    peak = pd.to_datetime(frame["peak_UTC"])
    year = peak.dt.year
    estimated = frame["goes_estimated_mean_class"].astype("string").str[0]
    strong = estimated.isin(["M", "X"])
    far_side = frame["visible_from_earth"] == False
    near_side = frame["visible_from_earth"] == True
    has_visibility = frame["visible_from_earth"].notna()
    counts = frame["4-10 keV"]
    goes = frame["GOES_class_time_of_flare"].astype("string")
    cpd = frame["cpd_filename"].astype("string")
    window = cpd.str.extract(CPD_WINDOW).apply(pd.to_datetime, format="%Y%m%dT%H%M%S")
    per_month = peak.dt.strftime("%Y-%m").value_counts()
    per_day = peak.dt.strftime("%Y-%m-%d").value_counts()
    sorted_peaks = peak.sort_values().reset_index(drop=True)
    gaps = sorted_peaks.diff()
    longest = gaps.idxmax()
    share_by_year = strong[year.between(2021, 2025)].groupby(year).mean()
    lat = frame["hgs_lat"].dropna()
    longitude_says_visible = frame["hgs_lon"].abs() <= 90

    return {
        "n_flares": len(frame),
        "n_columns": frame.shape[1],
        "n_columns_with_missing": int(frame.isna().any().sum()),
        "n_attenuator_flares": int(frame["att_in"].sum()),
        "n_imaging_errors": int(frame["error_with_imaging"].sum()),
        "max_counts_4_10_keV": int(counts.max()),
        "mean_solo_distance_au": float(frame["solo_position_AU_distance"].mean()),
        "n_strong_flares": int(strong.sum()),
        "n_x_class_flares": int((estimated == "X").sum()),
        "most_common_estimated_class": frame["goes_estimated_mean_class"].value_counts().idxmax(),
        "n_far_side_flares": int(far_side.sum()),
        "n_missing_goes_class": int(goes.isna().sum()),
        "median_counts_4_10_keV_attenuator": float(counts[frame["att_in"]].median()),
        "n_strong_far_side_flares": int((strong & far_side).sum()),
        "median_counts_4_10_keV_at_1au": float(
            (counts * frame["solo_position_AU_distance"] ** 2).median()
        ),
        "n_high_energy_below_background": int((frame["50-84 keV"] < frame["bkg 50-84 keV"]).sum()),
        "northern_share": float((lat > 0).mean()),
        "most_common_peak_hour": int(peak.dt.hour.value_counts().idxmax()),
        "median_duration_min": float((end - start).dt.total_seconds().median() / 60),
        "max_estimated_class_flux": float(class_to_flux(frame["goes_estimated_mean_class"]).max()),
        "median_goes_class_flux": float(class_to_flux(goes).median()),
        "median_goes_flux_recorded": float(frame["GOES_flux_time_of_flare"][goes.notna()].median()),
        "n_cpd_version_2": int(cpd.str.contains("_V02_").sum()),
        "median_cpd_window_min": float((window[1] - window[0]).dt.total_seconds().median() / 60),
        "flares_2024": int((year == 2024).sum()),
        "n_months": int(per_month.size),
        "busiest_month": per_month.idxmax(),
        "busiest_month_count": int(per_month.max()),
        "busiest_day": per_day.idxmax(),
        "busiest_day_count": int(per_day.max()),
        "attenuator_flares_2025": int(frame["att_in"][year == 2025].sum()),
        "median_rise_time_min": float((peak - start).dt.total_seconds().median() / 60),
        "strong_share_2024": float(strong[year == 2024].mean()),
        "year_highest_strong_share": int(share_by_year.idxmax()),
        "strong_share_far_side": float(strong[far_side].mean()),
        "n_visibility_longitude_mismatch": int(
            (near_side != longitude_says_visible)[has_visibility].sum()
        ),
        "goes_class_agreement": float((goes.str[0] == estimated)[near_side & goes.notna()].mean()),
        "median_waiting_time_min": float(gaps.dt.total_seconds().median() / 60),
        "longest_gap_days": float(gaps.max().total_seconds() / 86400),
        "longest_gap_start_day": sorted_peaks[longest - 1].strftime("%Y-%m-%d"),
        "spearman_counts_vs_estimated_flux": float(
            counts.rank().corr(frame["goes_estimated_mean_flux"].rank())
        ),
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
        print(f"{'ok ' if agrees else 'DIFF'} {claim_id:<34} {expected!s:<12} {actual}")
    unchecked = sorted(set(computed) - set(answers))
    if unchecked:
        print(f"computed but not in answers.yaml: {unchecked}")
    return int(bool(disagreements))


if __name__ == "__main__":
    sys.exit(main())
