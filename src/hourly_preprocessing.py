import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.aqi_calculator import calculate_aqi
except ImportError:
    from aqi_calculator import calculate_aqi


INPUT_FILE = Path("data/incoming/dehradun_hourly_raw.csv")
OUTPUT_FILE = Path("data/incoming/dehradun_hourly_processed.csv")
REPORT_FILE = Path("data/monitoring/hourly_data_quality.json")

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Ozone", "AT", "RH"
]

MODEL_FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH"
]


def preprocess_hourly_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Hourly data missing: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "station_id", "parameter_name", "period_start", "mean"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    original_rows = len(df)

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce"
    )

    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

    df["parameter_name"] = (
        df["parameter_name"].astype(str).str.strip()
    )

    invalid_rows = int(
        df[["period_start", "mean"]].isna().any(axis=1).sum()
    )

    df = df.dropna(subset=["period_start", "mean"]).copy()
    df = df[df["parameter_name"].isin(POLLUTANTS)].copy()

    if df.empty:
        raise ValueError("No valid hourly observations found.")

    # Preserve the station identity and avoid mixing stations.
    stations = df["station_id"].dropna().unique()

    if len(stations) != 1:
        raise ValueError(
            f"Expected exactly one station; found {len(stations)}. "
            "Do not combine readings from different stations."
        )

    duplicates = int(
        df.duplicated(
            ["station_id", "parameter_name", "period_start"]
        ).sum()
    )

    # Aggregate duplicate observations without fabricating values.
    pivot = df.pivot_table(
        index="period_start",
        columns="parameter_name",
        values="mean",
        aggfunc="mean"
    )

    pivot.columns.name = None

    full_index = pd.date_range(
        pivot.index.min().floor("h"),
        pivot.index.max().floor("h"),
        freq="h"
    )

    hourly = pivot.reindex(
        index=full_index,
        columns=POLLUTANTS
    )

    hourly.index.name = "period_start"
    hourly = hourly.reset_index()

    # Model feature completeness is separate from AQI validity.
    hourly["model_features_complete"] = (
        hourly[MODEL_FEATURES].notna().all(axis=1)
    )

    # Official-method hourly calculation:
    # 24-hour rolling averages for 24h pollutants,
    # maximum rolling 8-hour averages for CO and Ozone,
    # minimum valid-hour coverage and pollutant-count rule.
    result = calculate_aqi(
        hourly.copy(),
        mode="hourly",
        timestamp_col="period_start"
    )

    if "CPCB_AQI" not in result.columns:
        raise ValueError("Hourly calculator did not generate CPCB_AQI.")

    if "AQI_status" not in result.columns:
        raise ValueError("Hourly calculator did not generate AQI_status.")

    # An hour is AQI-valid only if the CPCB calculation produced a value.
    result["aqi_data_complete"] = result["CPCB_AQI"].notna()

    # Backward-compatible alias for existing anomaly modules.
    result["estimated_AQI"] = result["CPCB_AQI"]

    missing_counts = {
        pollutant: int(hourly[pollutant].isna().sum())
        for pollutant in POLLUTANTS
    }

    total_hours = len(result)
    complete_aqi = int(result["aqi_data_complete"].sum())
    complete_model = int(result["model_features_complete"].sum())

    status_counts = {
        str(key): int(value)
        for key, value in result["AQI_status"].value_counts(
            dropna=False
        ).items()
    }

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "station_id": str(stations[0]),
        "source_rows": original_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_readings_aggregated": duplicates,
        "hourly_timestamps": total_hours,
        "first_observation": str(result["period_start"].min()),
        "last_observation": str(result["period_start"].max()),
        "aqi_method": "CPCB rolling hourly calculation",
        "aqi_valid_hours": complete_aqi,
        "aqi_coverage_percent": round(
            complete_aqi / total_hours * 100, 2
        ) if total_hours else 0,
        "complete_model_feature_hours": complete_model,
        "model_feature_coverage_percent": round(
            complete_model / total_hours * 100, 2
        ) if total_hours else 0,
        "aqi_status_counts": status_counts,
        "missing_values_by_pollutant": missing_counts,
        "note": (
            "CPCB_AQI is calculated using rolling pollutant averages, "
            "minimum valid-hour coverage and the minimum pollutant rule. "
            "Missing measurements remain NaN. A missing hourly reading "
            "does not automatically invalidate the AQI if the required "
            "rolling-window coverage is still satisfied."
        )
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\n========== AEROAI CPCB HOURLY PREPROCESSING ==========")
    print("Station:", stations[0])
    print("Hourly observations:", total_hours)
    print("CPCB AQI valid hours:", complete_aqi)
    print("Complete model-feature hours:", complete_model)
    print("AQI coverage:", report["aqi_coverage_percent"], "%")

    print("\nAQI status counts:")
    print(pd.Series(status_counts).to_string())

    print("\nMissing values:")
    print(pd.Series(missing_counts).to_string())

    print("\nProcessed data saved:", OUTPUT_FILE)
    print("Quality report saved:", REPORT_FILE)


if __name__ == "__main__":
    preprocess_hourly_data()
