import os
import json
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = ROOT / "data/incoming/openaq_daily_raw.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_daily_quality.json"

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 317035
MIN_COVERAGE = 75.0
TZ = ZoneInfo("Asia/Kolkata")

SENSORS = {
    "PM2.5": (12237080, "µg/m³"),
    "PM10": (12237079, "µg/m³"),
    "NO2": (12237077, "ppb"),
    "SO2": (12237082, "ppb"),
    "CO": (12237075, "ppb"),
    "Ozone": (12237078, "µg/m³"),
    "AT": (12237083, "c"),
    "RH": (12237081, "%"),
}


def get_api_key():
    key = os.getenv("OPENAQ_API_KEY")

    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break

    if not key:
        raise RuntimeError("OPENAQ_API_KEY missing")

    return key


def collect():
    headers = {"X-API-Key": get_api_key()}
    all_data = {}

    print("\n========== OPENAQ DAILY COLLECTION ==========")

    for parameter, (sensor_id, expected_unit) in SENSORS.items():
        response = requests.get(
            f"{BASE_URL}/sensors/{sensor_id}/days",
            headers=headers,
            params={
                "date_from": "2026-10-27",
                "date_to": "2026-11-27",
                "limit": 100,
                "page": 1,
            },
            timeout=30,
        )
        response.raise_for_status()

        results = response.json().get("results", [])
        count = 0

        for item in results:
            period = item.get("period") or {}
            start = (period.get("datetimeFrom") or {}).get("local")

            if not start:
                continue

            date = datetime.fromisoformat(start).date().isoformat()

            coverage = item.get("coverage") or {}
            actual_unit = (item.get("parameter") or {}).get("units")

            record = {
                "value": item.get("value"),
                "unit": actual_unit,
                "coverage_percent": coverage.get("percentComplete"),
                "sensor_id": sensor_id,
            }

            all_data.setdefault(date, {})
            all_data[date][parameter] = record
            count += 1

        print(f"{parameter}: {count} daily records")

    rows = []

    for date, readings in sorted(all_data.items()):
        row = {"date": date}

        for parameter in SENSORS:
            reading = readings.get(parameter, {})

            row[f"{parameter}_value"] = reading.get("value")
            row[f"{parameter}_unit"] = reading.get("unit")
            row[f"{parameter}_coverage"] = reading.get(
                "coverage_percent"
            )
            row[f"{parameter}_sensor_id"] = reading.get("sensor_id")

        coverage_values = [
            row.get(f"{p}_coverage") for p in SENSORS
        ]

        row["all_values_present"] = all(
            pd.notna(row.get(f"{p}_value")) for p in SENSORS
        )

        row["all_coverages_available"] = all(
            pd.notna(value) for value in coverage_values
        )

        row["minimum_sensor_coverage"] = (
            min(coverage_values)
            if all(pd.notna(x) for x in coverage_values)
            else None
        )

        row["coverage_gate_pass"] = (
            row["all_values_present"]
            and row["all_coverages_available"]
            and row["minimum_sensor_coverage"] >= MIN_COVERAGE
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("OpenAQ returned no daily records.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary = OUTPUT_FILE.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(OUTPUT_FILE)

    report = {
        "source": "OpenAQ API v3",
        "location_id": LOCATION_ID,
        "station": "Doon University, Dehradun - UKPCB",
        "collected_at": datetime.now(TZ).isoformat(),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "records": len(df),
        "minimum_coverage_threshold_percent": MIN_COVERAGE,
        "coverage_gate_pass_dates": int(
            df["coverage_gate_pass"].sum()
        ),
        "coverage_gate_fail_dates": int(
            (~df["coverage_gate_pass"]).sum()
        ),
        "unit_review_required": {
            "CO": "OpenAQ ppb vs historical mg/m³; verify sensor scale",
            "NO2": "ppb requires validated conversion",
            "SO2": "ppb requires validated conversion",
            "AT": "OpenAQ reports c; verify equivalence to °C",
        },
        "model_integration": "NOT APPROVED",
        "note": (
            "Raw daily observations preserved. Coverage gate is not "
            "a guarantee of sensor accuracy or model compatibility."
        ),
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== COLLECTION RESULT ==========")
    print("Dates:", len(df))
    print("Coverage gate passed:", report["coverage_gate_pass_dates"])
    print("Coverage gate failed:", report["coverage_gate_fail_dates"])
    print("Saved dataset:", OUTPUT_FILE)
    print("Saved quality report:", REPORT_FILE)
    print("\nLatest rows:")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    collect()
