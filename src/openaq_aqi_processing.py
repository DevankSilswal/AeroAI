import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from src.aqi_calculator import calculate_aqi
except ModuleNotFoundError:
    from aqi_calculator import calculate_aqi

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data/incoming/openaq_hourly_raw.csv"
OUTPUT_FILE = ROOT / "data/processed/openaq_hourly_aqi.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_aqi_quality.json"

SENSORS = {
    12237080: ("PM2.5", "µg/m³"),
    12237079: ("PM10", "µg/m³"),
    12237078: ("Ozone", "µg/m³"),
    12237083: ("AT", "°C"),
    12237081: ("RH", "%"),
}


def normalize_unit(value):
    return (
        str(value).strip().lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("³", "3")
        .replace(" ", "")
        .replace("°", "")
    )


def main():
    print("\n========== OPENAQ CPCB-METHOD AQI PROCESSING ==========")

    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    raw = pd.read_csv(RAW_FILE)

    required = {
        "sensor_id", "parameter_name", "period_start",
        "period_end", "mean", "unit", "has_flags"
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(f"Missing raw data columns: {missing}")

    raw["sensor_id"] = pd.to_numeric(
        raw["sensor_id"], errors="coerce"
    )

    raw["mean"] = pd.to_numeric(
        raw["mean"], errors="coerce"
    )

    raw["period_start"] = pd.to_datetime(
        raw["period_start"], errors="coerce", utc=True
    )

    raw["period_end"] = pd.to_datetime(
        raw["period_end"], errors="coerce", utc=True
    )

    if raw[["sensor_id", "period_start", "period_end"]].isna().any().any():
        raise ValueError("Invalid sensor IDs or hourly timestamps found.")

    if raw["period_end"].duplicated().sum() == len(raw):
        raise ValueError("Invalid hourly timestamp data.")

    allowed_parameters = {
        parameter for parameter, _ in SENSORS.values()
    }

    raw = raw[raw["parameter_name"].isin(allowed_parameters)].copy()

    for sensor_id, (parameter, expected_unit) in SENSORS.items():
        subset = raw[raw["parameter_name"] == parameter]

        if subset.empty:
            raise ValueError(f"No readings found for {parameter}.")

        sensor_ids = set(subset["sensor_id"].astype(int).unique())

        if sensor_ids != {sensor_id}:
            raise ValueError(
                f"Unexpected sensor identity for {parameter}: {sensor_ids}"
            )

        observed_units = {
            normalize_unit(x)
            for x in subset["unit"].dropna().unique()
        }

        if observed_units != {normalize_unit(expected_unit)}:
            raise ValueError(
                f"Unit mismatch for {parameter}: {observed_units}"
            )

    # Never silently use flagged or invalid sensor measurements.
    raw["is_flagged"] = (
        raw["has_flags"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    flagged_count = int(raw["is_flagged"].sum())

    clean = raw[
        (~raw["is_flagged"])
        & raw["mean"].notna()
        & (raw["mean"] >= 0)
    ].copy()

    clean = clean.sort_values("period_end")

    duplicates = int(
        clean.duplicated(["sensor_id", "period_end"]).sum()
    )

    if duplicates:
        raise ValueError(
            f"Duplicate sensor-hour records detected: {duplicates}"
        )

    duration = clean["period_end"] - clean["period_start"]

    if not (duration == pd.Timedelta(hours=1)).all():
        raise ValueError("Unexpected hourly measurement duration.")

    minute_offsets = set(clean["period_end"].dt.minute)
    second_offsets = set(clean["period_end"].dt.second)

    if len(minute_offsets) != 1 or second_offsets != {0}:
        raise ValueError("Hourly timestamp alignment is inconsistent.")

    # Rebuild wide data from validated long-form measurements.
    pivot = clean.pivot_table(
        index="period_end",
        columns="parameter_name",
        values="mean",
        aggfunc="mean"
    )

    pivot.columns.name = None

    full_index = pd.date_range(
        start=pivot.index.min(),
        end=pivot.index.max(),
        freq="h"
    )

    pivot = pivot.reindex(full_index)
    pivot.index.name = "period_end"

    wide = pivot.reset_index()

    # OpenAQ timestamps mark the end of each hour.
    wide["period_start"] = (
        wide["period_end"] - pd.Timedelta(hours=1)
    )

    total_hours = len(wide)

    coverage = {}

    for parameter, expected_unit in SENSORS.values():
        valid_count = int(wide[parameter].notna().sum())

        coverage[parameter] = {
            "unit": expected_unit,
            "valid_hours": valid_count,
            "missing_hours": total_hours - valid_count,
            "coverage_percent": round(
                valid_count / total_hours * 100, 2
            )
        }

    print("Validated raw readings:", len(clean))
    print("Excluded flagged readings:", flagged_count)
    print("Duplicate records:", duplicates)
    print("Hourly timestamps:", total_hours)

    # Official calculator logic, applied to OpenAQ sensor measurements.
    result = calculate_aqi(
        wide.copy(),
        mode="hourly",
        timestamp_col="period_start"
    )

    valid = result["CPCB_AQI"].notna()
    valid_count = int(valid.sum())

    result["data_source"] = "OpenAQ API v3"
    result["aqi_provenance"] = (
        "CPCB-method estimate from OpenAQ sensor hourly measurements"
    )
    result["official_cpcb_aqi"] = False

    categories = {
        str(key): int(value)
        for key, value in
        result.loc[valid, "AQI_category"].value_counts().items()
    }

    if not valid.any():
        raise RuntimeError("No valid OpenAQ AQI observations available.")

    now = pd.Timestamp.now(tz="UTC")

    valid_period_ends = pd.to_datetime(
        result.loc[valid, "period_end"],
        errors="coerce",
        utc=True
    ).dropna()

    if valid_period_ends.empty:
        raise RuntimeError(
            "No valid measurement-end timestamps for OpenAQ AQI."
        )

    latest_valid_end = valid_period_ends.max()
    latest_end = wide["period_end"].max()

    # Freshness is based on the end of the latest valid AQI measurement.
    age_hours = (now - latest_valid_end).total_seconds() / 3600

    report = {
        "source": "OpenAQ API v3",
        "station": "Doon University, Dehradun - UKPCB",
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_file": str(RAW_FILE),
        "output_file": str(OUTPUT_FILE),
        "method": "CPCB breakpoints with rolling 24h and maximum rolling 8h averages",
        "official_cpcb_aqi": False,
        "historical_hours": total_hours,
        "validated_readings": int(len(clean)),
        "excluded_flagged_readings": flagged_count,
        "duplicate_sensor_records": duplicates,
        "first_period_end": wide["period_end"].min().isoformat(),
        "latest_period_end": latest_end.isoformat(),
        "latest_valid_aqi_period_end": latest_valid_end.isoformat(),
        "latest_data_age_hours": round(age_hours, 2),
        "valid_aqi_hours": valid_count,
        "insufficient_aqi_hours": int(total_hours - valid_count),
        "aqi_coverage_percent": round(
            valid_count / total_hours * 100, 2
        ),
        "category_distribution": categories,
        "sensor_coverage": coverage,
        "limitations": [
            "This is an estimate calculated from OpenAQ sensor readings, not an official CPCB station AQI.",
            "NO2, SO2, CO, NH3 and Pb are excluded from this dataset.",
            "Missing readings are not interpolated.",
            "Hourly AQI validity follows the existing calculator's minimum coverage and pollutant rules.",
            "Model training approval requires a separate daily dataset, target alignment and baseline evaluation."
        ],
        "model_integration": "NOT APPROVED"
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output_temp = OUTPUT_FILE.with_suffix(".csv.tmp")
    report_temp = REPORT_FILE.with_suffix(".json.tmp")

    result.to_csv(output_temp, index=False)
    report_temp.write_text(json.dumps(report, indent=4))

    output_temp.replace(OUTPUT_FILE)
    report_temp.replace(REPORT_FILE)

    print("\n========== AQI PROCESSING RESULT ==========")
    print("Valid AQI hours:", valid_count)
    print("Insufficient AQI hours:", total_hours - valid_count)
    print("AQI coverage:", report["aqi_coverage_percent"], "%")
    print("Latest valid AQI:", result.loc[valid, "CPCB_AQI"].iloc[-1])
    print("Latest category:", result.loc[valid, "AQI_category"].iloc[-1])
    print("Latest data age:", report["latest_data_age_hours"], "hours")

    print("\nCategory distribution:")
    print(pd.Series(categories).to_string())

    print("\nSaved AQI dataset:", OUTPUT_FILE)
    print("Saved quality report:", REPORT_FILE)
    print("\nOfficial CPCB AQI: NO")
    print("Model integration: NOT APPROVED")


if __name__ == "__main__":
    main()
