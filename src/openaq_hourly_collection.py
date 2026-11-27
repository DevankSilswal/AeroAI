"""Collect separate historical hourly measurements from OpenAQ."""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 317035
DAYS_BACK = 180
PAGE_LIMIT = 1000
INDIA_TZ = ZoneInfo("Asia/Kolkata")

RAW_FILE = ROOT / "data/incoming/openaq_hourly_raw.csv"
WIDE_FILE = ROOT / "data/incoming/openaq_hourly_wide.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_hourly_quality.json"

# Only sensors with currently verified compatible measurement units.
SENSORS = {
    12237080: {
        "parameter": "PM2.5",
        "unit": "µg/m³",
        "accepted_units": {"µg/m³", "μg/m³", "ug/m3", "µg/m3"},
    },
    12237079: {
        "parameter": "PM10",
        "unit": "µg/m³",
        "accepted_units": {"µg/m³", "μg/m³", "ug/m3", "µg/m3"},
    },
    12237078: {
        "parameter": "Ozone",
        "unit": "µg/m³",
        "accepted_units": {"µg/m³", "μg/m³", "ug/m3", "µg/m3"},
    },
    12237083: {
        "parameter": "AT",
        "unit": "°C",
        "accepted_units": {"c", "°c"},
    },
    12237081: {
        "parameter": "RH",
        "unit": "%",
        "accepted_units": {"%"},
    },
}


def get_api_key():
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise RuntimeError("OPENAQ_API_KEY missing from .env")
    return key


def normalize_unit(unit):
    return str(unit or "").strip().lower().replace("μ", "µ")


def fetch_sensor_hours(session, sensor_id, config, start_utc, end_utc):
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"

    params = {
        "datetime_from": start_utc.isoformat().replace("+00:00", "Z"),
        "datetime_to": end_utc.isoformat().replace("+00:00", "Z"),
        "limit": PAGE_LIMIT,
        "page": 1,
    }

    records = []
    page = 1

    while True:
        params["page"] = page

        # Retry transient API/network failures without treating stale data as fresh.
        response = None
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=60,
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    break

                if attempt == max_attempts:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = min(float(retry_after), 30.0) if retry_after else 2 ** attempt
                except (TypeError, ValueError):
                    wait_seconds = 2 ** attempt

                print(
                    f"OpenAQ transient HTTP {response.status_code} for "
                    f"sensor {sensor_id}, page {page}; "
                    f"retry {attempt}/{max_attempts - 1} in {wait_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(wait_seconds)

            except requests.RequestException as exc:
                if attempt == max_attempts:
                    raise
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"OpenAQ request error for sensor {sensor_id}, page {page}: "
                    f"{type(exc).__name__}; retry {attempt}/{max_attempts - 1} "
                    f"in {wait_seconds}s",
                    flush=True,
                )
                time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError(
                f"No response from OpenAQ for sensor {sensor_id}, page {page}"
            )
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results", [])
        meta = payload.get("meta", {})

        if not results:
            break

        for item in results:
            parameter = item.get("parameter") or {}
            actual_unit = parameter.get("units")

            if normalize_unit(actual_unit) not in {
                normalize_unit(x) for x in config["accepted_units"]
            }:
                raise ValueError(
                    f"Unit mismatch for {config['parameter']} "
                    f"(sensor {sensor_id}): {actual_unit}"
                )

            period = item.get("period") or {}
            end_info = period.get("datetimeTo") or {}
            start_info = period.get("datetimeFrom") or {}

            end_value = end_info.get("local") or end_info.get("utc")
            start_value = start_info.get("local") or start_info.get("utc")

            if not end_value:
                continue

            period_end = pd.Timestamp(end_value)

            if period_end.tzinfo is None:
                period_end = period_end.tz_localize(INDIA_TZ)
            else:
                period_end = period_end.tz_convert(INDIA_TZ)

            if start_value:
                period_start = pd.Timestamp(start_value)

                if period_start.tzinfo is None:
                    period_start = period_start.tz_localize(INDIA_TZ)
                else:
                    period_start = period_start.tz_convert(INDIA_TZ)
            else:
                period_start = period_end - pd.Timedelta(hours=1)

            records.append({
                "station_id": LOCATION_ID,
                "sensor_id": sensor_id,
                "parameter_name": config["parameter"],
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "mean": item.get("value"),
                "unit": actual_unit,
                "coverage_percent": (
                    (item.get("coverage") or {}).get("percentComplete")
                ),
                "has_flags": (
                    (item.get("flagInfo") or {}).get("hasFlags")
                ),
            })

        # OpenAQ may return meta.found as ">1000" instead of an integer.
        # Continue until an empty or short page confirms the final page.
        if len(results) < PAGE_LIMIT:
            break

        page += 1
        time.sleep(0.1)

    if not records:
        raise RuntimeError(
            f"No hourly measurements returned for "
            f"{config['parameter']} sensor {sensor_id}"
        )

    return records


def save_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def collect():
    print("\n========== OPENAQ HISTORICAL HOURLY COLLECTION ==========")

    api_key = get_api_key()
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=DAYS_BACK)

    print("Station location ID:", LOCATION_ID)
    print("Requested history:", DAYS_BACK, "days")
    print("From UTC:", start_utc.isoformat())
    print("To UTC:", end_utc.isoformat())

    session = requests.Session()
    session.headers.update({"X-API-Key": api_key})

    all_records = []

    # Fetch all required sensors before changing any output file.
    for sensor_id, config in SENSORS.items():
        print(
            f"\nFetching {config['parameter']} "
            f"(sensor {sensor_id})...",
            flush=True
        )

        records = fetch_sensor_hours(
            session,
            sensor_id,
            config,
            start_utc,
            end_utc,
        )

        all_records.extend(records)
        print("Hourly records:", len(records))

    raw = pd.DataFrame(all_records)

    if raw.empty:
        raise RuntimeError("OpenAQ returned no historical records.")

    raw["period_end"] = pd.to_datetime(raw["period_end"], utc=True)
    raw["period_start"] = pd.to_datetime(raw["period_start"], utc=True)
    raw["mean"] = pd.to_numeric(raw["mean"], errors="coerce")
    raw["coverage_percent"] = pd.to_numeric(
        raw["coverage_percent"], errors="coerce"
    )

    # Preserve the previous separate OpenAQ history and upsert new records.
    if RAW_FILE.exists():
        old = pd.read_csv(RAW_FILE)
        if not old.empty:
            old["period_end"] = pd.to_datetime(
                old["period_end"], errors="coerce", utc=True
            )
            old["period_start"] = pd.to_datetime(
                old["period_start"], errors="coerce", utc=True
            )
            old["mean"] = pd.to_numeric(old["mean"], errors="coerce")
            old["coverage_percent"] = pd.to_numeric(
                old["coverage_percent"], errors="coerce"
            )
            raw = pd.concat([old, raw], ignore_index=True)

    raw = raw.dropna(subset=["period_end"])
    raw = raw.drop_duplicates(
        subset=["sensor_id", "period_end"],
        keep="last"
    )
    raw = raw.sort_values(["period_end", "parameter_name"])

    # Build a separate wide dataset. Missing sensor-hours remain NaN.
    wide = raw.pivot_table(
        index="period_end",
        columns="parameter_name",
        values="mean",
        aggfunc="mean",
    )

    wide.columns.name = None

    # Preserve OpenAQ's actual hourly boundary alignment.
    # Do not floor :30 UTC timestamps to :00, which would lose readings.
    minute_offsets = set(wide.index.minute)
    second_offsets = set(wide.index.second)

    if len(minute_offsets) != 1 or second_offsets != {0}:
        raise ValueError(
            "Unexpected hourly timestamp alignment; "
            "refusing to reindex and potentially lose observations."
        )

    full_index = pd.date_range(
        start=wide.index.min(),
        end=wide.index.max(),
        freq="h",
    )

    wide = wide.reindex(full_index)
    wide.index.name = "period_end"
    wide = wide.reset_index()

    # OpenAQ timestamps mark the END of the measured hour.
    wide["period_start"] = (
        pd.to_datetime(wide["period_end"], utc=True)
        - pd.Timedelta(hours=1)
    )

    # Retain source identity and unit documentation.
    wide["source"] = "OpenAQ API v3"
    wide["location_id"] = LOCATION_ID

    expected_hours = DAYS_BACK * 24
    per_sensor = {}

    for sensor_id, config in SENSORS.items():
        subset = raw[raw["sensor_id"] == sensor_id]
        parameter = config["parameter"]
        valid = subset.dropna(subset=["mean"])

        unique_hours = int(
            subset["period_end"].nunique()
        )

        coverage_values = subset["coverage_percent"].dropna()

        per_sensor[parameter] = {
            "sensor_id": sensor_id,
            "expected_unit": config["unit"],
            "observed_unit": sorted(
                subset["unit"].dropna().astype(str).unique().tolist()
            ),
            "records": int(len(subset)),
            "valid_values": int(len(valid)),
            "missing_values": int(subset["mean"].isna().sum()),
            "unique_hours": unique_hours,
            "window_hours": expected_hours,
            "window_coverage_percent": round(
                unique_hours / expected_hours * 100, 2
            ),
            "average_reported_coverage_percent": (
                round(float(coverage_values.mean()), 2)
                if not coverage_values.empty else None
            ),
            "first_period_end": (
                subset["period_end"].min().isoformat()
                if not subset.empty else None
            ),
            "last_period_end": (
                subset["period_end"].max().isoformat()
                if not subset.empty else None
            ),
            "flagged_records": int(
                subset["has_flags"].fillna(False).astype(bool).sum()
            ),
        }

    report = {
        "source": "OpenAQ API v3 /sensors/{id}/hours",
        "station": "Doon University, Dehradun - UKPCB",
        "location_id": LOCATION_ID,
        "collected_at": datetime.now(INDIA_TZ).isoformat(),
        "requested_days": DAYS_BACK,
        "requested_start_utc": start_utc.isoformat(),
        "requested_end_utc": end_utc.isoformat(),
        "raw_records": int(len(raw)),
        "wide_hourly_rows": int(len(wide)),
        "parameters": per_sensor,
        "model_integration": "NOT APPROVED - separate validation required",
        "aqi_calculation": "NOT YET RUN",
        "notes": [
            "OpenAQ hourly values are hourly means.",
            "OpenAQ hourly timestamps represent the end of each hour.",
            "Missing measurements remain NaN; no interpolation is applied.",
            "Only PM2.5, PM10, Ozone, temperature and RH are collected.",
            "NO2, SO2 and CO are excluded pending unit verification.",
            "This dataset is separate from the original XKDR training data.",
            "Historical source readings are not automatically official CPCB AQI.",
        ],
    }

    # Write both temporary files first, then replace the dedicated outputs.
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw_temp = RAW_FILE.with_suffix(".csv.tmp")
    wide_temp = WIDE_FILE.with_suffix(".csv.tmp")
    report_temp = REPORT_FILE.with_suffix(".json.tmp")

    raw.to_csv(raw_temp, index=False)
    wide.to_csv(wide_temp, index=False)
    report_temp.write_text(json.dumps(report, indent=4))

    raw_temp.replace(RAW_FILE)
    wide_temp.replace(WIDE_FILE)
    report_temp.replace(REPORT_FILE)

    print("\n========== OPENAQ HOURLY RESULT ==========")
    print("Long-form records:", len(raw))
    print("Wide hourly rows:", len(wide))
    print("First period end:", raw["period_end"].min())
    print("Latest period end:", raw["period_end"].max())

    for parameter, details in per_sensor.items():
        print(
            f"{parameter}: {details['valid_values']} valid values | "
            f"{details['window_coverage_percent']}% window coverage"
        )

    print("\nSaved raw data:", RAW_FILE)
    print("Saved wide data:", WIDE_FILE)
    print("Saved quality report:", REPORT_FILE)
    print("\nModel integration remains NOT APPROVED.")


if __name__ == "__main__":
    collect()
