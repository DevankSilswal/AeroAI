import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

API_URL = "https://api.dharun.dev/v1/air-quality"

LATITUDE = 30.269444
LONGITUDE = 78.044167

OUTPUT_JSON = ROOT / "data/incoming/current_aqi_snapshot.json"
OUTPUT_CSV = ROOT / "data/incoming/nearby_cpcb_stations.csv"

INDIA_TZ = ZoneInfo("Asia/Kolkata")
STALE_AFTER_HOURS = 3


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=4))
    temporary.replace(path)


def collect_live_air_quality():

    print("\n========== AEROAI CURRENT AIR QUALITY ==========")

    response = requests.get(
        API_URL,
        params={"near": f"{LATITUDE},{LONGITUDE}"},
        timeout=30
    )

    response.raise_for_status()
    payload = response.json()

    stations = payload.get("data", [])
    metadata = payload.get("meta", {})

    if not stations:
        raise ValueError("API returned no monitoring stations.")

    # Prefer Doon University station specifically.
    station = next(
        (
            item for item in stations
            if "doon university" in item.get("station", "").lower()
        ),
        None
    )

    if station is None:
        raise ValueError(
            "Doon University station was not found in the API response. "
            "Existing snapshot has been preserved."
        )

    if station.get("aqi") is None or station.get("updated") is None:
        raise ValueError("Required AQI value or update timestamp is missing.")

    updated = pd.Timestamp(station["updated"])

    if updated.tzinfo is None:
        updated = updated.tz_localize(INDIA_TZ)
    else:
        updated = updated.tz_convert(INDIA_TZ)

    now = datetime.now(INDIA_TZ)
    age_hours = (now - updated.to_pydatetime()).total_seconds() / 3600

    if age_hours < 0:
        raise ValueError("Station timestamp is unexpectedly in the future.")

    freshness = (
        "Fresh"
        if age_hours <= STALE_AFTER_HOURS
        else "Stale"
    )

    pollutant_subindices = {}

    for name, values in station.get("pollutants", {}).items():
        pollutant_subindices[name] = {
            "minimum": values.get("min"),
            "maximum": values.get("max"),
            "average_subindex": values.get("avg")
        }

    nearby = []

    for item in stations:
        nearby.append({
            "station": item.get("station"),
            "city": item.get("city"),
            "state": item.get("state"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "aqi": item.get("aqi"),
            "category": item.get("category"),
            "distance_km": item.get("distance_km"),
            "updated": item.get("updated"),
            "main_pollutant": item.get("main_pollutant")
        })

    snapshot = {
        "source": "CPCB monitoring data via India Air Quality API",
        "source_url": metadata.get("source", {}).get("url"),
        "license": metadata.get("source", {}).get("licence"),
        "attribution": metadata.get("source", {}).get("attribution"),
        "retrieved_at": now.isoformat(timespec="seconds"),
        "station_name": station.get("station"),
        "city": station.get("city"),
        "state": station.get("state"),
        "latitude": station.get("latitude"),
        "longitude": station.get("longitude"),
        "aqi": station.get("aqi"),
        "category": station.get("category"),
        "main_pollutant": station.get("main_pollutant"),
        "station_updated_at": updated.isoformat(),
        "station_data_age_hours": round(age_hours, 2),
        "freshness": freshness,
        "pollutant_subindices": pollutant_subindices,
        "nearby_stations": nearby,
        "note": (
            "AQI is supplied by the API using CPCB station data. "
            "Individual pollutant values are sub-indices, not raw "
            "concentrations. Feed freshness reflects the station timestamp."
        )
    }

    station_table = pd.DataFrame(nearby)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV and JSON atomically.
    temporary_csv = OUTPUT_CSV.with_suffix(".csv.tmp")
    station_table.to_csv(temporary_csv, index=False)
    temporary_csv.replace(OUTPUT_CSV)

    save_json_atomic(OUTPUT_JSON, snapshot)

    print("Station:", snapshot["station_name"])
    print("Current CPCB AQI:", snapshot["aqi"])
    print("Category:", snapshot["category"])
    print("Updated:", snapshot["station_updated_at"])
    print("Data age:", snapshot["station_data_age_hours"], "hours")
    print("Freshness:", freshness)
    print("Nearby stations:", len(nearby))
    print("Saved:", OUTPUT_JSON)
    print("Saved:", OUTPUT_CSV)

    print("\nAttribution:", snapshot["attribution"])


if __name__ == "__main__":
    try:
        collect_live_air_quality()
    except Exception as error:
        print("Current AQI collection failed:", error)
        raise SystemExit(1)
