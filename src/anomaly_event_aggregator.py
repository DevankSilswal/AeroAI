import json
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/monitoring/hourly_anomaly_report.csv")
OUTPUT_FILE = Path("data/monitoring/hourly_anomaly_events.csv")
SUMMARY_FILE = Path("data/monitoring/hourly_anomaly_events.json")

PRIMARY_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]
AQI_FEATURE = "CPCB_AQI"


def aggregate_anomaly_events():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)
    required = [
        "period_start", "feature", "observed_value",
        "baseline_median", "robust_z_score", "status"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing anomaly report columns: {missing}")

    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")

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

            per_hour = supporting.groupby("period_start")["feature"].nunique()
            peak = group.loc[group["observed_value"].idxmax()]
            confirmed_hours = int((per_hour >= 2).sum())

            max_aqi = float(peak["observed_value"])
            max_score = group["robust_z_score"].max()

            events.append({
                "episode_id": int(episode_id) + 1,
                "start_time": timestamps.min().isoformat(),
                "end_time": timestamps.max().isoformat(),
                "aqi_metric": AQI_FEATURE,
                "aqi_alert_hours": int(len(group)),
                "max_CPCB_AQI": max_aqi,
                "max_estimated_AQI": max_aqi,
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
                    "Multi-pollutant statistical anomaly"
                    if confirmed_hours > 0
                    else "CPCB AQI statistical anomaly"
                )
            })

    result = pd.DataFrame(events)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    summary = {
        "aqi_metric": AQI_FEATURE,
        "total_feature_alerts": int(len(alerts)),
        "CPCB_AQI_alert_hours": int(len(aqi_alerts)),
        "estimated_aqi_alert_hours": int(len(aqi_alerts)),
        "unique_hours_with_any_feature_alert": int(
            alerts["period_start"].nunique()
        ),
        "pollution_episodes": len(events),
        "episodes": events,
        "warning": (
            "Statistical anomalies are not confirmed real-world events. "
            "Event causes require independent evidence."
        )
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=4))

    print("\\n========== AEROAI POLLUTION EPISODES ==========")
    print("AQI metric:", AQI_FEATURE)
    print("Total feature alerts:", len(alerts))
    print("CPCB AQI alert hours:", len(aqi_alerts))
    print("Unique alert hours:", alerts["period_start"].nunique())
    print("Grouped pollution episodes:", len(events))
    if not result.empty:
        print(result.to_string(index=False))
    print("Saved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)


if __name__ == "__main__":
    aggregate_anomaly_events()
