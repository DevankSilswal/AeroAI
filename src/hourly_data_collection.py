import os
import io
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=ROOT / ".env")

API_KEY = os.getenv("AQI_API_KEY")

BASE_URL = "https://airquality.xkdr.org/v1/measurements"

STATION_ID = "site_5594"

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Ozone", "AT", "RH"
]

OUTPUT_FILE = ROOT / "data/incoming/dehradun_hourly_raw.csv"


def collect_hourly_data():

    if not API_KEY:
        raise ValueError("AQI_API_KEY missing in .env")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    params = [
        ("station", STATION_ID),
        ("start", start_date.isoformat()),
        ("end", end_date.isoformat()),
        ("agg", "hourly"),
        ("format", "csv"),
    ]

    for pollutant in POLLUTANTS:
        params.append(("parameter", pollutant))

    print("\n========== AEROAI HOURLY DATA COLLECTION ==========")
    print("Date range:", start_date, "to", end_date)

    response = requests.get(
        BASE_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        params=params,
        timeout=90
    )

    response.raise_for_status()

    if not response.text.strip():
        print("API returned an empty response. Existing data preserved.")
        return

    new_data = pd.read_csv(io.StringIO(response.text))

    if new_data.empty:
        print("No new hourly observations. Existing data preserved.")
        return

    required = {
        "station_id", "parameter_name",
        "period_start", "mean"
    }

    missing = required - set(new_data.columns)

    if missing:
        raise ValueError(f"Unexpected API schema. Missing: {missing}")

    new_data["period_start"] = pd.to_datetime(
        new_data["period_start"], errors="coerce"
    )

    new_data["mean"] = pd.to_numeric(
        new_data["mean"], errors="coerce"
    )

    new_data["parameter_name"] = (
        new_data["parameter_name"].astype(str).str.strip()
    )

    new_data = new_data[
        new_data["station_id"].astype(str) == STATION_ID
    ]

    new_data = new_data[
        new_data["parameter_name"].isin(POLLUTANTS)
    ]

    new_data = new_data.dropna(
        subset=["period_start"]
    )

    if new_data.empty:
        raise ValueError("No valid hourly observations in API response.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        old_data = pd.read_csv(OUTPUT_FILE)

        old_data["period_start"] = pd.to_datetime(
            old_data["period_start"], errors="coerce"
        )

        combined = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )
    else:
        combined = new_data.copy()

    combined = combined.dropna(
        subset=["period_start"]
    )

    combined = combined.drop_duplicates(
        subset=[
            "station_id",
            "parameter_name",
            "period_start"
        ],
        keep="last"
    )

    combined = combined.sort_values(
        ["period_start", "parameter_name"]
    )

    # Write a temporary file first to protect the existing dataset.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    combined.to_csv(temporary_file, index=False)
    temporary_file.replace(OUTPUT_FILE)

    print("New API records:", len(new_data))
    print("Total stored records:", len(combined))
    print("Unique timestamps:", combined["period_start"].nunique())
    print("Latest observation:", combined["period_start"].max())
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    collect_hourly_data()
