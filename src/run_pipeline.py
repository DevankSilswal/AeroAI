from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    ("CPCB Current AQI Refresh", "src/live_air_quality.py"),
    ("OpenAQ Live Sensor Refresh", "src/openaq_data.py"),
    ("Daily Data Collection", "src/data_collection.py"),
    ("Hourly Data Collection", "src/hourly_data_collection.py"),
    ("Daily Preprocessing", "src/preprocessing.py"),
    ("Hourly Preprocessing", "src/hourly_preprocessing.py"),
    ("AQI Calculation", "src/aqi_calculator.py"),
    ("AQI Prediction", "src/prediction.py"),
    ("Daily Data Drift Detection", "src/drift_detection.py"),
    ("Prediction Performance Monitoring", "src/model_monitoring.py"),
    ("Hourly Anomaly Detection", "src/hourly_anomaly_detection.py"),
    ("Pollution Episode Aggregation", "src/anomaly_event_aggregator.py"),
    ("Intelligent Retraining Decision", "src/retraining_decision_engine.py"),
]

OPTIONAL_SOURCE_STEPS = {
    "CPCB Current AQI Refresh",
    "OpenAQ Live Sensor Refresh",
    "Daily Data Collection",
    "Hourly Data Collection",
}

CPCB_SNAPSHOT_FILE = ROOT / "data/incoming/current_aqi_snapshot.json"
OPENAQ_SNAPSHOT_FILE = ROOT / "data/incoming/openaq_snapshot.json"
DAILY_DATA_FILE = ROOT / "data/incoming/dehradun_latest.csv"
HOURLY_DATA_FILE = ROOT / "data/incoming/dehradun_hourly_raw.csv"

DECISION_FILE = ROOT / "data/monitoring/retraining_decision.json"
PREDICTION_FRESHNESS_FILE = ROOT / "data/monitoring/prediction_freshness.json"
PIPELINE_REPORT_FILE = ROOT / "data/monitoring/pipeline_status.json"

CPCB_MAX_AGE_HOURS = 3
OPENAQ_MAX_AGE_HOURS = 3
DAILY_MAX_AGE_HOURS = 48
HOURLY_MAX_AGE_HOURS = 3


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def age_in_hours(timestamp):
    parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)

    if pd.isna(parsed):
        return None

    return (
        pd.Timestamp.now(tz="UTC") - parsed
    ).total_seconds() / 3600


def latest_csv_timestamp(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)

    timestamp_col = next(
        (
            col for col in
            ["period_start", "date", "timestamp", "datetime"]
            if col in df.columns
        ),
        None
    )

    if timestamp_col is None:
        return None

    timestamps = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True
    ).dropna()

    if timestamps.empty:
        return None

    return timestamps.max()


def validate_step_output(name):
    if name == "CPCB Current AQI Refresh":
        snapshot = read_json(CPCB_SNAPSHOT_FILE)

        if not snapshot or snapshot.get("aqi") is None:
            return "SOURCE_UNAVAILABLE"

        age = age_in_hours(snapshot.get("station_updated_at"))

        if age is None or age < 0 or age > CPCB_MAX_AGE_HOURS:
            return "STALE_DATA"

        return "SUCCESS"

    if name == "OpenAQ Live Sensor Refresh":
        snapshot = read_json(OPENAQ_SNAPSHOT_FILE)

        if not snapshot:
            return "SOURCE_UNAVAILABLE"

        readings = snapshot.get("readings", {})

        # PM sensors are required for the live monitoring check.
        # Other pollutant units are not automatically approved for training.
        for parameter in ["pm25", "pm10"]:
            reading = readings.get(parameter)

            if not reading:
                return "PARTIAL_DATA"

            age = age_in_hours(reading.get("timestamp"))

            if age is None or age < 0:
                return "PARTIAL_DATA"

            if age > OPENAQ_MAX_AGE_HOURS:
                return "STALE_DATA"

        return "SUCCESS"

    if name == "Daily Data Collection":
        latest = latest_csv_timestamp(DAILY_DATA_FILE)

        if latest is None:
            return "SOURCE_UNAVAILABLE"

        age = age_in_hours(latest)

        if age is None or age < 0 or age > DAILY_MAX_AGE_HOURS:
            return "STALE_DATA"

        return "SUCCESS"

    if name == "Hourly Data Collection":
        latest = latest_csv_timestamp(HOURLY_DATA_FILE)

        if latest is None:
            return "SOURCE_UNAVAILABLE"

        age = age_in_hours(latest)

        if age is None or age < 0 or age > HOURLY_MAX_AGE_HOURS:
            return "STALE_DATA"

        return "SUCCESS"

    if name == "AQI Prediction":
        report = read_json(PREDICTION_FRESHNESS_FILE)

        if not report:
            return "FAILED"

        status = report.get("status")

        if status == "BLOCKED_STALE_DATA":
            return "BLOCKED - stale data"

        if status != "FRESH":
            return "FAILED"

        return "SUCCESS"

    return "SUCCESS"


def execute_step(name, script):
    print(f"\n========== {name} ==========", flush=True)

    try:
        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=True
        )

        status = validate_step_output(name)

        if status == "SUCCESS":
            print(f"{name}: SUCCESS", flush=True)
        else:
            print(f"{name}: {status}", flush=True)

        return status

    except subprocess.CalledProcessError as error:
        if name in OPTIONAL_SOURCE_STEPS:
            print(
                f"{name}: SOURCE_UNAVAILABLE "
                f"(exit code {error.returncode})",
                flush=True
            )
            return "SOURCE_UNAVAILABLE"

        print(
            f"{name}: FAILED (exit code {error.returncode})",
            flush=True
        )
        return "FAILED"

    except Exception as error:
        if name in OPTIONAL_SOURCE_STEPS:
            print(f"{name}: SOURCE_UNAVAILABLE ({error})", flush=True)
            return "SOURCE_UNAVAILABLE"

        print(f"{name}: FAILED ({error})", flush=True)
        return "FAILED"


def save_pipeline_report(results, final_status, started):
    report = {
        "pipeline": "AeroAI Adaptive AI Pipeline",
        "started_at": started.astimezone().isoformat(timespec="seconds"),
        "completed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "final_status": final_status,
        "stages": [
            {"stage": name, "status": status}
            for name, status in results
        ],
        "important_note": (
            "Fresh CPCB station AQI and OpenAQ PM sensor snapshots "
            "are for live monitoring. OpenAQ pollutant unit validation "
            "and historical model integration remain separate gates."
        ),
    }

    PIPELINE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary = PIPELINE_REPORT_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=4))
    temporary.replace(PIPELINE_REPORT_FILE)


def run_pipeline():
    started = datetime.now().astimezone()
    results = []

    print("\n==========================================")
    print("       AEROAI ADAPTIVE AI PIPELINE")
    print("==========================================")
    print("Started:", started.isoformat(timespec="seconds"))

    for name, script in STEPS:
        status = execute_step(name, script)
        results.append((name, status))

        if status == "FAILED":
            print("\nPipeline stopped because a core stage failed.")
            break

        if status != "SUCCESS":
            print(
                "Continuing with remaining stages. "
                "This source or stage is not fully healthy."
            )

    else:
        try:
            decision_report = read_json(DECISION_FILE)

            if not decision_report:
                raise FileNotFoundError(
                    "Retraining decision report was not generated."
                )

            decision = decision_report.get("decision")

            print("\n========== CONDITIONAL RETRAINING ==========")
            print("Decision:", decision)

            if decision == "CANDIDATE_TRAINING_ELIGIBLE":
                status = execute_step(
                    "Intelligent Candidate Training and Validation",
                    "src/intelligent_retraining.py"
                )

                results.append((
                    "Candidate Training and Validation",
                    status
                ))

            elif decision in [
                "WAIT_FOR_LABELLED_DATA",
                "MONITOR_AND_COLLECT_DATA"
            ]:
                results.append((
                    "Candidate Training and Validation",
                    "SKIPPED - retraining conditions not satisfied"
                ))

                print(
                    "Candidate training skipped. "
                    "Existing champion model remains active."
                )

            else:
                raise ValueError(
                    f"Unknown retraining decision: {decision}"
                )

        except Exception as error:
            print("Retraining decision processing failed:", error)
            results.append(("Conditional Retraining", "FAILED"))

    print("\n========== AEROAI PIPELINE SUMMARY ==========")

    for name, status in results:
        print(f"{name}: {status}")

    failed = any(
        status == "FAILED"
        for _, status in results
    )

    degraded = any(
        status != "SUCCESS" and not status.startswith("SKIPPED")
        for _, status in results
    )

    if failed:
        final_status = "FAILED"
        exit_code = 1
    elif degraded:
        final_status = "PARTIAL_SUCCESS"
        exit_code = 2
    else:
        final_status = "SUCCESS"
        exit_code = 0

    save_pipeline_report(results, final_status, started)

    print("\nPipeline result:", final_status)
    print("Pipeline report:", PIPELINE_REPORT_FILE)

    return exit_code


if __name__ == "__main__":
    sys.exit(run_pipeline())
