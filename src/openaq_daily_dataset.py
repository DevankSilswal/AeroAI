import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data/processed/openaq_hourly_aqi.csv"
OUTPUT = ROOT / "data/processed/openaq_next_day_dataset.csv"
REPORT = ROOT / "data/monitoring/openaq_daily_dataset_quality.json"

FEATURES = [
    "pm25_mean",
    "pm25_max",
    "pm10_mean",
    "pm10_max",
    "ozone_mean",
    "ozone_max",
    "temperature_mean",
    "humidity_mean",
    "day_end_aqi",
    "day_end_pm25_aqi",
    "day_end_pm10_aqi",
    "day_end_ozone_aqi",
]

MIN_LABELLED_ROWS = 90
LOCAL_TZ = "Asia/Kolkata"


def main():
    print("\n========== OPENAQ DAILY FORECAST DATASET ==========")

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = [
        "period_start", "PM2.5", "PM10", "Ozone",
        "AT", "RH", "CPCB_AQI",
        "PM2.5_AQI", "PM10_AQI", "Ozone_AQI",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce", utc=True
    )

    df = df.dropna(subset=["period_start"]).copy()

    for col in required:
        if col != "period_start":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["period_start"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps detected.")

    # Convert timestamps to Dehradun local dates.
    df["local_time"] = df["period_start"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["local_time"].dt.strftime("%Y-%m-%d")

    latest_date = df["local_date"].max()

    # Exclude the currently incomplete local calendar day.
    complete_window = df[df["local_date"] < latest_date].copy()

    if complete_window.empty:
        raise ValueError("No completed local calendar days available.")

    print("Latest source date:", latest_date)
    print("Excluded partial date:", latest_date)

    daily_rows = []

    for date, group in complete_window.groupby("local_date", sort=True):
        group = group.sort_values("period_start")

        counts = {
            "pm25_hours": int(group["PM2.5"].notna().sum()),
            "pm10_hours": int(group["PM10"].notna().sum()),
            "ozone_hours": int(group["Ozone"].notna().sum()),
            "temperature_hours": int(group["AT"].notna().sum()),
            "humidity_hours": int(group["RH"].notna().sum()),
            "valid_aqi_hours": int(group["CPCB_AQI"].notna().sum()),
        }

        enough_coverage = (
            counts["pm25_hours"] >= 16
            and counts["pm10_hours"] >= 16
            and counts["ozone_hours"] >= 8
            and counts["temperature_hours"] >= 12
            and counts["humidity_hours"] >= 12
            and counts["valid_aqi_hours"] >= 16
        )

        valid_aqi = group.dropna(subset=["CPCB_AQI"])

        last_valid = (
            valid_aqi.iloc[-1]
            if not valid_aqi.empty else None
        )

        row = {
            "date": date,
            **counts,
            "coverage_pass": bool(enough_coverage),
        }

        for column, name in [
            ("PM2.5", "pm25"),
            ("PM10", "pm10"),
            ("Ozone", "ozone"),
        ]:
            row[f"{name}_mean"] = group[column].mean()
            row[f"{name}_max"] = group[column].max()

        row["temperature_mean"] = group["AT"].mean()
        row["humidity_mean"] = group["RH"].mean()

        if last_valid is not None:
            row["day_end_aqi"] = last_valid["CPCB_AQI"]
            row["day_end_pm25_aqi"] = last_valid["PM2.5_AQI"]
            row["day_end_pm10_aqi"] = last_valid["PM10_AQI"]
            row["day_end_ozone_aqi"] = last_valid["Ozone_AQI"]
            row["label_timestamp"] = last_valid["period_start"].isoformat()
        else:
            row["day_end_aqi"] = np.nan
            row["day_end_pm25_aqi"] = np.nan
            row["day_end_pm10_aqi"] = np.nan
            row["day_end_ozone_aqi"] = np.nan
            row["label_timestamp"] = None

        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    daily["date"] = pd.to_datetime(daily["date"])

    # Reindex to calendar days so shift(-1) always means the next date.
    calendar = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D"
    )

    daily = (
        daily.set_index("date")
        .reindex(calendar)
    )
    daily.index.name = "date"
    daily = daily.reset_index()

    daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
    daily["next_day_cpcb_method_estimated_AQI"] = (
        daily["day_end_aqi"].shift(-1)
    )

    next_day_coverage = daily["coverage_pass"].shift(-1).fillna(False)

    source_features_valid = (
        daily["coverage_pass"].fillna(False)
        & daily[FEATURES].notna().all(axis=1)
    )

    target_valid = (
        daily["next_day_cpcb_method_estimated_AQI"].notna()
        & next_day_coverage
    )

    dataset = daily[
        source_features_valid & target_valid
    ].copy()

    dataset["date"] = dataset["date"].dt.strftime("%Y-%m-%d")
    dataset["target_date"] = dataset["target_date"].dt.strftime("%Y-%m-%d")

    if dataset.empty:
        raise RuntimeError("No valid next-day training rows produced.")

    dataset["target_name"] = (
        "Next-day OpenAQ CPCB-method estimated AQI"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    temp_output = OUTPUT.with_suffix(".csv.tmp")
    temp_report = REPORT.with_suffix(".json.tmp")

    dataset.to_csv(temp_output, index=False)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "OpenAQ historical hourly measurements",
        "input_file": str(INPUT),
        "output_file": str(OUTPUT),
        "target": "Next-day OpenAQ CPCB-method estimated AQI",
        "target_definition": (
            "Last valid hourly CPCB-method estimate on the next "
            "local calendar date"
        ),
        "feature_definition": (
            "Same-day hourly pollutant/weather summaries and "
            "last valid same-day AQI snapshot"
        ),
        "timezone": LOCAL_TZ,
        "excluded_partial_date": latest_date,
        "calendar_days": int(len(daily)),
        "coverage_pass_days": int(daily["coverage_pass"].fillna(False).sum()),
        "labelled_next_day_rows": int(len(dataset)),
        "minimum_required_rows": MIN_LABELLED_ROWS,
        "training_data_gate_pass": bool(
            len(dataset) >= MIN_LABELLED_ROWS
        ),
        "features": FEATURES,
        "first_source_date": dataset["date"].min(),
        "last_source_date": dataset["date"].max(),
        "last_target_date": dataset["target_date"].max(),
        "official_cpcb_aqi": False,
        "model_training_approved": False,
        "limitations": [
            "Target is an OpenAQ-based CPCB-method estimate, not official CPCB AQI.",
            "Only validated PM2.5, PM10 and Ozone sub-indices contribute to AQI.",
            "Missing hours are not interpolated.",
            "The partial latest local date is excluded.",
            "A chronological holdout and persistence baseline comparison are required before model promotion."
        ]
    }

    temp_report.write_text(json.dumps(report, indent=4))
    temp_output.replace(OUTPUT)
    temp_report.replace(REPORT)

    print("\n========== DATASET RESULT ==========")
    print("Calendar days:", report["calendar_days"])
    print("Coverage-pass days:", report["coverage_pass_days"])
    print("Next-day labelled rows:", report["labelled_next_day_rows"])
    print("Minimum required:", MIN_LABELLED_ROWS)
    print("Training data gate:", report["training_data_gate_pass"])
    print("Date range:", report["first_source_date"], "to", report["last_target_date"])
    print("Target:", report["target"])
    print("Official CPCB AQI:", report["official_cpcb_aqi"])
    print("Saved dataset:", OUTPUT)
    print("Saved report:", REPORT)


if __name__ == "__main__":
    main()
