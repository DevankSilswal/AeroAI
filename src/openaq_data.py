"""OpenAQ live air-quality collector for AeroAI."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "incoming" / "openaq_snapshot.json"

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 317035
INDIA_TZ = ZoneInfo("Asia/Kolkata")

# Only verified current sensor IDs; old 2022 sensors are excluded.
# Parameter names and units verified using OpenAQ sensor metadata.
SENSORS = {
    12237077: ("no2", "ppb"),
    12237081: ("relativehumidity", "%"),
    12237078: ("o3", "µg/m³"),
    12237083: ("temperature", "°C"),
    12237079: ("pm10", "µg/m³"),
    12237075: ("co", "ppb"),
    12237082: ("so2", "ppb"),
    12237080: ("pm25", "µg/m³"),
    12237076: ("no", "ppb"),
    14341196: ("nox", "ppb"),
    14341198: ("wind_speed", "m/s"),
    14341197: ("wind_direction", "deg"),
}

STALE_AFTER_HOURS = 3


def save_json_atomic(data):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=4))
    temporary.replace(OUTPUT_FILE)


def collect_openaq():
    api_key = os.getenv("OPENAQ_API_KEY")

    # Read .env without printing or exposing the key.
    if not api_key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY is missing.")

    response = requests.get(
        f"{BASE_URL}/locations/{LOCATION_ID}/latest",
        headers={"X-API-Key": api_key},
        params={"limit": 100},
        timeout=30,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    now = datetime.now(INDIA_TZ)
    readings = {}

    for item in results:
        sensor_id = item.get("sensorsId")

        if sensor_id not in SENSORS:
            continue

        parameter, verified_unit = SENSORS[sensor_id]
        timestamp = (item.get("datetime") or {}).get("local")

        if not timestamp:
            continue

        measured_at = datetime.fromisoformat(timestamp)
        if measured_at.tzinfo is None:
            measured_at = measured_at.replace(tzinfo=INDIA_TZ)
        else:
            measured_at = measured_at.astimezone(INDIA_TZ)

        age_hours = (now - measured_at).total_seconds() / 3600

        if age_hours < 0:
            continue

        value = item.get("value")
        if value is None:
            continue

        # Preserve only the newest observation for each parameter.
        previous = readings.get(parameter)
        if previous and previous["timestamp"] >= measured_at.isoformat():
            continue

        readings[parameter] = {
            "value": value,
            "unit": verified_unit,
            "timestamp": measured_at.isoformat(),
            "age_hours": round(age_hours, 2),
            "freshness": (
                "Fresh" if age_hours <= STALE_AFTER_HOURS else "Stale"
            ),
            "sensor_id": sensor_id,
        }

    snapshot = {
        "source": "OpenAQ API v3",
        "source_url": "https://explore.openaq.org/locations/317035",
        "station_name": "Doon University, Dehradun - UKPCB",
        "location_id": LOCATION_ID,
        "latitude": 30.269444,
        "longitude": 78.044167,
        "retrieved_at": now.isoformat(timespec="seconds"),
        "readings": readings,
        "fresh_reading_count": sum(
            x["freshness"] == "Fresh" for x in readings.values()
        ),
        "stale_reading_count": sum(
            x["freshness"] == "Stale" for x in readings.values()
        ),
        "note": (
            "Raw sensor measurements from OpenAQ. "
            "Not an official CPCB AQI value. "
            "Readings retain their original units and timestamps."
        ),
    }

    save_json_atomic(snapshot)

    print("\n========== AEROAI OPENAQ COLLECTION ==========")
    print("Station:", snapshot["station_name"])
    print("Parameters:", len(readings))
    print("Fresh readings:", snapshot["fresh_reading_count"])
    print("Stale readings:", snapshot["stale_reading_count"])

    for name, reading in sorted(readings.items()):
        print(
            f"{name}: {reading['value']} {reading['unit']} "
            f"| {reading['freshness']} "
            f"| {reading['timestamp']}"
        )

    print("\nSaved:", OUTPUT_FILE)
    return snapshot


if __name__ == "__main__":
    try:
        collect_openaq()
    except Exception as error:
        print("OpenAQ collection failed:", error)
        raise SystemExit(1)
