import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("data/incoming/dehradun_hourly_processed.csv")
OUTPUT_FILE = Path("data/monitoring/hourly_anomaly_report.csv")
SUMMARY_FILE = Path("data/monitoring/hourly_anomaly_summary.json")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CPCB_AQI"]
WINDOW_HOURS = 24
MINIMUM_BASELINE = 12
SPIKE_THRESHOLD = 3.5


def median_absolute_deviation(values):
    median = np.median(values)
    return np.median(np.abs(values - median))


def detect_hourly_anomalies():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Processed hourly data missing: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    missing = [c for c in ["period_start"] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.dropna(subset=["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)

    records = []

    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        previous = values.shift(1)

        rolling = previous.rolling(
            window=WINDOW_HOURS,
            min_periods=MINIMUM_BASELINE
        )

        baseline = rolling.median()
        mad = rolling.apply(median_absolute_deviation, raw=True)
        q25 = rolling.quantile(0.25)
        q75 = rolling.quantile(0.75)

        mad_scale = 1.4826 * mad
        iqr_scale = (q75 - q25) / 1.349
        scale = mad_scale.where(mad_scale > 0, iqr_scale)

        baseline_count = rolling.count()
        robust_score = (values - baseline) / scale

        for i in range(len(df)):
            observed = values.iloc[i]
            median = baseline.iloc[i]
            score = robust_score.iloc[i]
            count_value = baseline_count.iloc[i]
            count = int(count_value) if pd.notna(count_value) else 0
            current_scale = scale.iloc[i]

            if pd.isna(observed):
                status = "Missing observation"
            elif count < MINIMUM_BASELINE or pd.isna(median):
                status = "Insufficient baseline"
            elif pd.isna(current_scale) or current_scale <= 0:
                status = "Baseline variability unavailable"
            elif pd.notna(score) and score >= SPIKE_THRESHOLD:
                status = "Potential pollution spike"
            else:
                status = "Normal"

            records.append({
                "period_start": df["period_start"].iloc[i].isoformat(),
                "feature": feature,
                "observed_value": float(observed) if pd.notna(observed) else None,
                "baseline_median": float(median) if pd.notna(median) else None,
                "baseline_observations": count,
                "robust_z_score": round(float(score), 3) if pd.notna(score) else None,
                "status": status
            })

    report = pd.DataFrame(records)
    alerts = report[report["status"] == "Potential pollution spike"].copy()

    spike_counts = {
        feature: int((alerts["feature"] == feature).sum())
        for feature in FEATURES
    }

    summary = {
        "method": "Causal rolling median and MAD",
        "aqi_metric": "CPCB_AQI",
        "baseline_window_hours": WINDOW_HOURS,
        "minimum_baseline_observations": MINIMUM_BASELINE,
        "spike_threshold": SPIKE_THRESHOLD,
        "total_feature_observations": int(len(report)),
        "potential_spike_alerts": int(len(alerts)),
        "spikes_by_feature": spike_counts,
        "first_observation": str(df["period_start"].min()),
        "last_observation": str(df["period_start"].max()),
        "interpretation": (
            "Potential spikes are statistical alerts, not confirmed "
            "pollution events. Validate against data quality, weather "
            "and independent observations."
        ),
        "limitation": (
            "This detector identifies unusual increases after measurements "
            "arrive. It does not predict upcoming festivals or events."
        )
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=4))

    print("\\n========== AEROAI HOURLY ANOMALY DETECTION ==========")
    print("Features monitored:", len(FEATURES))
    print("AQI metric: CPCB_AQI")
    print("Potential spike alerts:", len(alerts))
    print("Alerts by feature:")
    print(pd.Series(spike_counts).to_string())
    print("Report saved:", OUTPUT_FILE)
    print("Summary saved:", SUMMARY_FILE)


if __name__ == "__main__":
    detect_hourly_anomalies()
