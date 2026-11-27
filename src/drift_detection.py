import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from aqi_calculator import FEATURES

REFERENCE_FILE = Path("data/raw/dehradun_aqi_dataset.csv")
CURRENT_FILE = Path("data/incoming/dehradun_aqi_processed.csv")

OUTPUT_CSV = Path("data/monitoring/drift_report.csv")
OUTPUT_JSON = Path("data/monitoring/drift_summary.json")


def calculate_psi(reference, current, bins=10):

    reference = pd.Series(
        pd.to_numeric(reference, errors="coerce")
    ).dropna().to_numpy()

    current = pd.Series(
        pd.to_numeric(current, errors="coerce")
    ).dropna().to_numpy()

    if len(reference) < 20 or len(current) < 5:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Smoothing avoids division by zero.
    epsilon = 0.0001
    ref_pct = np.maximum(ref_pct, epsilon)
    cur_pct = np.maximum(cur_pct, epsilon)

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def detect_drift():

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    missing_reference = [
        col for col in FEATURES if col not in reference.columns
    ]
    missing_current = [
        col for col in FEATURES if col not in current.columns
    ]

    if missing_reference or missing_current:
        raise ValueError(
            f"Missing reference features: {missing_reference}; "
            f"Missing current features: {missing_current}"
        )

    results = []

    for feature in FEATURES:

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        if np.isnan(psi):
            status = "Insufficient data"
        elif psi < 0.10:
            status = "Low drift"
        elif psi < 0.20:
            status = "Moderate drift"
        else:
            status = "High drift"

        results.append({
            "feature": feature,
            "PSI": round(psi, 4) if not np.isnan(psi) else None,
            "status": status
        })

    report = pd.DataFrame(results)

    valid = report.dropna(subset=["PSI"])

    if valid.empty:
        overall = "Insufficient data"
    elif (valid["PSI"] >= 0.20).any():
        overall = "High drift detected"
    elif (valid["PSI"] >= 0.10).any():
        overall = "Moderate drift detected"
    else:
        overall = "Low drift detected"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "reference_dataset": str(REFERENCE_FILE),
        "current_dataset": str(CURRENT_FILE),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "overall_status": overall,
        "interpretation": (
            "PSI is a distribution-shift indicator, not a direct "
            "measure of prediction accuracy. Current sample is small."
        ),
        "results": results
    }

    with open(OUTPUT_JSON, "w") as file:
        json.dump(summary, file, indent=4)

    print("\n========== DATA DRIFT REPORT ==========")
    print(report.to_string(index=False))
    print("\nOverall status:", overall)
    print("Reference rows:", len(reference))
    print("Current rows:", len(current))
    print("Saved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_JSON)
    print("\nNote: 13 current observations are too few for "
          "a strong drift conclusion.")


if __name__ == "__main__":
    try:
        detect_drift()
    except Exception as error:
        print("Drift detection failed:", error)
