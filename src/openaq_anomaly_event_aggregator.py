import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data/monitoring/openaq_hourly_anomaly_report.csv"
OUTPUT_FILE = ROOT / "data/monitoring/openaq_hourly_anomaly_events.csv"
SUMMARY_FILE = ROOT / "data/monitoring/openaq_hourly_anomaly_events.json"

PRIMARY_POLLUTANTS = ["PM2.5", "PM10", "Ozone"]
AQI_FEATURE = "OpenAQ_CPCB_method_AQI"

EVENT_COLUMNS = [
    "event_id",
    "start_time",
    "end_time",
    "duration_hours",
    "aqi_metric",
    "aqi_alert_hours",
    "maximum_openaq_method_aqi",
    "baseline_at_peak",
    "maximum_robust_score",
    "pollutants_flagged",
    "multi_pollutant_alert_hours",
    "interpretation",
]


def aggregate_anomaly_events():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "period_start",
        "feature",
        "observed_value",
        "baseline_median",
        "robust_z_score",
        "status",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing OpenAQ anomaly columns: {missing}")

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce", utc=True
    )

    alerts = df[
        df["status"].eq("Potential pollution spike")
    ].dropna(subset=["period_start"]).copy()

    aqi_alerts = alerts[
        alerts["feature"].eq(AQI_FEATURE)
    ].sort_values("period_start").copy()

    events = []

    if not aqi_alerts.empty:
        aqi_alerts["episode"] = (
            aqi_alerts["period_start"]
            .diff()
            .gt(pd.Timedelta(hours=1))
            .cumsum()
        )

        for episode_id, group in aqi_alerts.groupby("episode"):
            timestamps = group["period_start"]

            supporting = alerts[
                alerts["feature"].isin(PRIMARY_POLLUTANTS)
                & alerts["period_start"].isin(timestamps)
            ]

            per_hour = supporting.groupby(
                "period_start"
            )["feature"].nunique()

            peak = group.loc[group["observed_value"].idxmax()]

            confirmed_hours = int((per_hour >= 2).sum())
            max_aqi = float(peak["observed_value"])
            max_score = group["robust_z_score"].max()

            duration = (
                timestamps.max() - timestamps.min()
            ).total_seconds() / 3600 + 1

            events.append({
                "event_id": int(episode_id) + 1,
                "start_time": timestamps.min().isoformat(),
                "end_time": timestamps.max().isoformat(),
                "duration_hours": round(duration, 2),
                "aqi_metric": AQI_FEATURE,
                "aqi_alert_hours": int(len(group)),
                "maximum_openaq_method_aqi": max_aqi,
                "baseline_at_peak": (
                    float(peak["baseline_median"])
                    if pd.notna(peak["baseline_median"]) else None
                ),
                "maximum_robust_score": (
                    float(max_score) if pd.notna(max_score) else None
                ),
                "pollutants_flagged": sorted(
                    supporting["feature"].unique().tolist()
                ),
                "multi_pollutant_alert_hours": confirmed_hours,
                "interpretation": (
                    "Multi-pollutant OpenAQ statistical anomaly"
                    if confirmed_hours > 0
                    else "OpenAQ AQI-method statistical anomaly"
                ),
            })

    result = pd.DataFrame(events, columns=EVENT_COLUMNS)

    summary = {
        "source": "OpenAQ API v3",
        "station": "Doon University, Dehradun - UKPCB",
        "aqi_metric": AQI_FEATURE,
        "official_cpcb_aqi": False,
        "total_feature_alerts": int(len(alerts)),
        "openaq_method_aqi_alert_hours": int(len(aqi_alerts)),
        "unique_hours_with_any_feature_alert": int(
            alerts["period_start"].nunique()
        ),
        "pollution_episodes": len(events),
        "episodes": events,
        "warning": (
            "These are statistical anomalies from OpenAQ sensor data, "
            "not confirmed real-world pollution events or official "
            "CPCB AQI alerts. Event causes require independent evidence."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output_temp = OUTPUT_FILE.with_suffix(".csv.tmp")
    summary_temp = SUMMARY_FILE.with_suffix(".json.tmp")

    result.to_csv(output_temp, index=False)
    summary_temp.write_text(json.dumps(summary, indent=4))

    output_temp.replace(OUTPUT_FILE)
    summary_temp.replace(SUMMARY_FILE)

    print("\n========== OPENAQ POLLUTION EPISODES ==========")
    print("AQI metric:", AQI_FEATURE)
    print("Official CPCB AQI: NO")
    print("Total feature alerts:", len(alerts))
    print("OpenAQ-method AQI alert hours:", len(aqi_alerts))
    print("Grouped OpenAQ episodes:", len(events))

    if not result.empty:
        print(result.to_string(index=False))

    print("Saved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)


if __name__ == "__main__":
    aggregate_anomaly_events()
