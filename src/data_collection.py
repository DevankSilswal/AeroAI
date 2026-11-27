import os
import io
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AQI_API_KEY")
BASE_URL = "https://airquality.xkdr.org/v1/measurements"

STATION_ID = "site_5594"

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Ozone", "AT", "RH"
]

OUTPUT_FILE = Path("data/incoming/dehradun_latest.csv")


def collect_data():
    if not API_KEY:
        raise ValueError("AQI_API_KEY missing. Check your .env file.")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    params = [
        ("station", STATION_ID),
        ("start", start_date.isoformat()),
        ("end", end_date.isoformat()),
        ("agg", "daily"),
        ("format", "csv"),
    ]

    for pollutant in POLLUTANTS:
        params.append(("parameter", pollutant))

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    print("Fetching Dehradun air-quality data...")
    print(f"Date range: {start_date} to {end_date}")

    response = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=90
    )

    response.raise_for_status()

    if not response.text.strip():
        print("API returned an empty response.")
        return

    new_data = pd.read_csv(io.StringIO(response.text))

    if new_data.empty:
        print("No new readings available for this period.")
        return

    required = {
        "station_id", "parameter_name",
        "period_start", "mean"
    }

    missing = required - set(new_data.columns)

    if missing:
        raise ValueError(f"Unexpected API response. Missing: {missing}")

    new_data["period_start"] = pd.to_datetime(
        new_data["period_start"], errors="coerce"
    )

    new_data["mean"] = pd.to_numeric(
        new_data["mean"], errors="coerce"
    )

    new_data = new_data.dropna(
        subset=["period_start", "mean"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        old_data = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )
    else:
        combined = new_data.copy()

    combined["period_start"] = pd.to_datetime(
        combined["period_start"], errors="coerce"
    )

    combined = combined.drop_duplicates(
        subset=["station_id", "parameter_name", "period_start"],
        keep="last"
    )

    combined = combined.sort_values(
        ["period_start", "parameter_name"]
    )

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\n========== DATA COLLECTION RESULT ==========")
    print("New API rows:", len(new_data))
    print("Total stored rows:", len(combined))
    print("Pollutants:", sorted(combined["parameter_name"].unique()))
    print("Latest date:", combined["period_start"].max())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    try:
        collect_data()
    except requests.exceptions.RequestException as error:
        print("API request failed:", error)
    except Exception as error:
        print("Data collection failed:", error)
