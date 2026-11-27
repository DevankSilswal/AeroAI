import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data/processed/openaq_hourly_aqi.csv"
OUTPUT_FILE = ROOT / "data/monitoring/openaq_hourly_anomaly_report.csv"
SUMMARY_FILE = ROOT / "data/monitoring/openaq_hourly_anomaly_summary.json"

FEATURES = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "Ozone": "Ozone",
    "OpenAQ_CPCB_method_AQI": "CPCB_AQI",
}

WINDOW_HOURS = 24
MINIMUM_BASELINE = 12
SPIKE_THRESHOLD = 3.5


def median_absolute_deviation(values):
    median = np.median(values)
    return np.median(np.abs(values - median))


def detect_hourly_anomalies():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = ["period_start"] + list(FEATURES.values())
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing OpenAQ AQI columns: {missing}")

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce", utc=True
    )

    if df["period_start"].isna().any():
        raise ValueError("Invalid OpenAQ timestamps found.")

    df = df.sort_values("period_start").reset_index(drop=True)

    if df["period_start"].duplicated().any():
        raise ValueError("Duplicate OpenAQ hourly timestamps found.")

    records = []

    for feature, column in FEATURES.items():
        values = pd.to_numeric(df[column], errors="coerce")
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
                "observed_value": (
                    float(observed) if pd.notna(observed) else None
                ),
                "baseline_median": (
                    float(median) if pd.notna(median) else None
                ),
                "baseline_observations": count,
                "robust_z_score": (
                    round(float(score), 3) if pd.notna(score) else None
                ),
                "status": status,
            })

    report = pd.DataFrame(records)

    alerts = report[
        report["status"] == "Potential pollution spike"
    ]

    summary = {
        "source": "OpenAQ API v3",
        "station": "Doon University, Dehradun - UKPCB",
        "aqi_metric": "OpenAQ_CPCB_method_AQI",
        "official_cpcb_aqi": False,
        "aqi_provenance": (
            "CPCB-method estimate calculated from OpenAQ hourly sensors"
        ),
        "method": "Causal rolling median and MAD",
        "baseline_window_hours": WINDOW_HOURS,
        "minimum_baseline_observations": MINIMUM_BASELINE,
        "spike_threshold": SPIKE_THRESHOLD,
        "features_monitored": list(FEATURES),
        "total_feature_observations": int(len(report)),
        "potential_spike_alerts": int(len(alerts)),
        "alerts_by_feature": {
            feature: int((alerts["feature"] == feature).sum())
            for feature in FEATURES
        },
        "first_observation": (
            str(df["period_start"].min()) if not df.empty else None
        ),
        "last_observation": (
            str(df["period_start"].max()) if not df.empty else None
        ),
        "interpretation": (
            "Statistical alerts from OpenAQ readings; not confirmed "
            "pollution events or official CPCB AQI alerts."
        ),
        "limitation": (
            "This detector identifies unusual increases after measurements "
            "arrive. It does not predict upcoming festivals or events."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output_temp = OUTPUT_FILE.with_suffix(".csv.tmp")
    summary_temp = SUMMARY_FILE.with_suffix(".json.tmp")

    report.to_csv(output_temp, index=False)
    summary_temp.write_text(json.dumps(summary, indent=4))

    output_temp.replace(OUTPUT_FILE)
    summary_temp.replace(SUMMARY_FILE)

    print("\n========== OPENAQ HOURLY ANOMALY DETECTION ==========")
    print("Source: OpenAQ API v3")
    print("Official CPCB AQI: NO")
    print("Features monitored:", len(FEATURES))
    print("Potential spike alerts:", len(alerts))
    print("Alerts by feature:")
    print(pd.Series(summary["alerts_by_feature"]).to_string())
    print("Report saved:", OUTPUT_FILE)
    print("Summary saved:", SUMMARY_FILE)


if __name__ == "__main__":
    detect_hourly_anomalies()
