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
    ("OpenAQ Historical Hourly Refresh", "src/openaq_hourly_collection.py"),
    ("OpenAQ Hourly AQI Processing", "src/openaq_aqi_processing.py"),
    ("OpenAQ Daily Forecast Dataset", "src/openaq_daily_dataset.py"),
    ("OpenAQ Demo Retraining and Forecast", "src/openaq_demo_workflow.py"),
    ("OpenAQ Hourly Anomaly Detection", "src/openaq_hourly_anomaly_detection.py"),
    ("OpenAQ Pollution Episode Aggregation", "src/openaq_anomaly_event_aggregator.py"),
]

OPTIONAL_SOURCE_STEPS = {
    "CPCB Current AQI Refresh",
    "OpenAQ Live Sensor Refresh",
    "Daily Data Collection",
    "Hourly Data Collection",
    "OpenAQ Historical Hourly Refresh",
}

CPCB_SNAPSHOT_FILE = ROOT / "data/incoming/current_aqi_snapshot.json"
OPENAQ_SNAPSHOT_FILE = ROOT / "data/incoming/openaq_snapshot.json"
DAILY_DATA_FILE = ROOT / "data/incoming/dehradun_latest.csv"
HOURLY_DATA_FILE = ROOT / "data/incoming/dehradun_hourly_raw.csv"
OPENAQ_HOURLY_RAW_FILE = ROOT / "data/incoming/openaq_hourly_raw.csv"
OPENAQ_HOURLY_AQI_FILE = ROOT / "data/processed/openaq_hourly_aqi.csv"
OPENAQ_DAILY_FORECAST_FILE = ROOT / "data/processed/openaq_next_day_dataset.csv"
OPENAQ_DAILY_QUALITY_FILE = ROOT / "data/monitoring/openaq_daily_dataset_quality.json"
OPENAQ_DEMO_REPORT_FILE = ROOT / "data/monitoring/openaq_demo_workflow.json"
OPENAQ_HOURLY_MAX_AGE_HOURS = 3

DECISION_FILE = ROOT / "data/monitoring/retraining_decision.json"
PREDICTION_FRESHNESS_FILE = ROOT / "data/monitoring/prediction_freshness.json"
PIPELINE_REPORT_FILE = ROOT / "data/monitoring/pipeline_status.json"

CPCB_MAX_AGE_HOURS = 3
OPENAQ_MAX_AGE_HOURS = 3
DAILY_MAX_AGE_HOURS = 48
HOURLY_MAX_AGE_HOURS = 3


# A stage runs only when all its required upstream stages succeeded.
STAGE_DEPENDENCIES = {
    "Daily Preprocessing": ["Daily Data Collection"],
    "Hourly Preprocessing": ["Hourly Data Collection"],
    "AQI Calculation": ["Daily Preprocessing"],
    "AQI Prediction": ["Daily Data Collection", "Daily Preprocessing", "AQI Calculation"],
    "Daily Data Drift Detection": ["Daily Data Collection", "Daily Preprocessing"],
    "Prediction Performance Monitoring": ["Daily Data Collection", "AQI Prediction"],
    "Hourly Anomaly Detection": ["Hourly Data Collection", "Hourly Preprocessing"],
    "Pollution Episode Aggregation": ["Hourly Anomaly Detection"],
    "Intelligent Retraining Decision": [
        "Daily Data Collection",
        "Daily Preprocessing",
        "Hourly Data Collection",
        "Hourly Preprocessing",
        "AQI Calculation",
        "AQI Prediction",
        "Prediction Performance Monitoring",
        "Hourly Anomaly Detection",
        "Pollution Episode Aggregation",
    ],
    "OpenAQ Hourly AQI Processing": ["OpenAQ Historical Hourly Refresh"],
    "OpenAQ Daily Forecast Dataset": ["OpenAQ Hourly AQI Processing"],
    "OpenAQ Demo Retraining and Forecast": ["OpenAQ Daily Forecast Dataset"],
    "OpenAQ Hourly Anomaly Detection": ["OpenAQ Hourly AQI Processing"],
    "OpenAQ Pollution Episode Aggregation": ["OpenAQ Hourly Anomaly Detection"],
}


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

    if name == "OpenAQ Historical Hourly Refresh":
        if not OPENAQ_HOURLY_RAW_FILE.exists():
            return "SOURCE_UNAVAILABLE"

        try:
            raw = pd.read_csv(OPENAQ_HOURLY_RAW_FILE)
        except Exception:
            return "SOURCE_UNAVAILABLE"

        required = {"sensor_id", "period_end", "mean", "has_flags"}
        if not required.issubset(raw.columns):
            return "SOURCE_UNAVAILABLE"

        raw["sensor_id"] = pd.to_numeric(
            raw["sensor_id"], errors="coerce"
        )
        raw["mean"] = pd.to_numeric(raw["mean"], errors="coerce")
        raw["period_end"] = pd.to_datetime(
            raw["period_end"], errors="coerce", utc=True
        )

        flagged = (
            raw["has_flags"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        required_sensors = [12237080, 12237079, 12237078]

        for sensor_id in required_sensors:
            readings = raw[
                (raw["sensor_id"] == sensor_id)
                & (~flagged)
                & raw["mean"].notna()
                & (raw["mean"] >= 0)
                & raw["period_end"].notna()
            ]

            if readings.empty:
                return "SOURCE_UNAVAILABLE"

            latest = readings["period_end"].max()
            age = age_in_hours(latest)

            if age is None or age < 0:
                return "PARTIAL_DATA"

            if age > OPENAQ_HOURLY_MAX_AGE_HOURS:
                return "STALE_DATA"

        return "SUCCESS"

    if name == "OpenAQ Hourly AQI Processing":
        if not OPENAQ_HOURLY_AQI_FILE.exists():
            return "SOURCE_UNAVAILABLE"

        try:
            result = pd.read_csv(OPENAQ_HOURLY_AQI_FILE)
        except Exception:
            return "SOURCE_UNAVAILABLE"

        required = {
            "period_end", "CPCB_AQI", "official_cpcb_aqi"
        }
        if not required.issubset(result.columns):
            return "SOURCE_UNAVAILABLE"

        official = (
            result["official_cpcb_aqi"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        if official.any():
            return "FAILED"

        result["CPCB_AQI"] = pd.to_numeric(
            result["CPCB_AQI"], errors="coerce"
        )
        result["period_end"] = pd.to_datetime(
            result["period_end"], errors="coerce", utc=True
        )

        # Only valid AQI rows and their measurement-end timestamps count.
        valid = result.dropna(
            subset=["CPCB_AQI", "period_end"]
        )

        if valid.empty:
            return "INSUFFICIENT_DATA"

        latest = valid["period_end"].max()
        age = age_in_hours(latest)

        if age is None or age < 0:
            return "PARTIAL_DATA"

        if age > OPENAQ_HOURLY_MAX_AGE_HOURS:
            return "STALE_DATA"

        return "SUCCESS"

    if name == "OpenAQ Daily Forecast Dataset":
        if not OPENAQ_DAILY_FORECAST_FILE.exists():
            return "SOURCE_UNAVAILABLE"

        quality = read_json(OPENAQ_DAILY_QUALITY_FILE)
        if not quality:
            return "SOURCE_UNAVAILABLE"

        if quality.get("training_data_gate_pass") is not True:
            return "INSUFFICIENT_DATA"

        return "SUCCESS"

    if name == "OpenAQ Demo Retraining and Forecast":
        report = read_json(OPENAQ_DEMO_REPORT_FILE)
        if not report:
            return "SOURCE_UNAVAILABLE"

        if report.get("workflow_status") != "COMPLETED":
            return "FAILED"

        if not report.get("forecast"):
            return "INSUFFICIENT_DATA"

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
    print(f"::group::{name}", flush=True)
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

        print("::endgroup::", flush=True)
        return status

    except subprocess.CalledProcessError as error:
        if name in OPTIONAL_SOURCE_STEPS:
            print(
                f"{name}: SOURCE_UNAVAILABLE "
                f"(exit code {error.returncode})",
                flush=True
            )
            print("::endgroup::", flush=True)
            return "SOURCE_UNAVAILABLE"

        print(
            f"{name}: FAILED (exit code {error.returncode})",
            flush=True
        )
        print("::endgroup::", flush=True)
        return "FAILED"

    except Exception as error:
        if name in OPTIONAL_SOURCE_STEPS:
            print(f"{name}: SOURCE_UNAVAILABLE ({error})", flush=True)
            print("::endgroup::", flush=True)
            return "SOURCE_UNAVAILABLE"

        print(f"{name}: FAILED ({error})", flush=True)
        print("::endgroup::", flush=True)
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
    stage_status = {}

    print("\\n==========================================")
    print("       AEROAI ADAPTIVE AI PIPELINE")
    print("==========================================")
    print("Started:", started.isoformat(timespec="seconds"))

    for name, script in STEPS:
        dependencies = STAGE_DEPENDENCIES.get(name, [])
        blocked = [
            dependency
            for dependency in dependencies
            if stage_status.get(dependency) != "SUCCESS"
        ]

        if blocked:
            print(f"::group::{name}", flush=True)
            blocked_statuses = [
                stage_status.get(dependency, "NOT_EXECUTED")
                for dependency in blocked
            ]

            upstream_failure = any(
                status == "FAILED" or status == "SKIPPED_UPSTREAM_FAILURE"
                for status in blocked_statuses
            )

            status = (
                "SKIPPED_UPSTREAM_FAILURE"
                if upstream_failure
                else "SKIPPED_STALE_INPUT"
            )

            print(f"\\n========== {name} ==========", flush=True)
            print(
                f"{name}: {status} | Blocked by: {', '.join(blocked)}",
                flush=True
            )
            print("::endgroup::", flush=True)
        else:
            status = execute_step(name, script)

        results.append((name, status))
        stage_status[name] = status

        if status == "FAILED":
            print("\\nPipeline stopped because a core stage failed.")
            break

        if status != "SUCCESS":
            print(
                "This stage or one of its required inputs is not healthy.",
                flush=True
            )

    else:
        decision_status = stage_status.get(
            "Intelligent Retraining Decision"
        )

        print("::group::Conditional Candidate Training and Validation", flush=True)
        print("\\n========== CONDITIONAL RETRAINING ==========")

        if decision_status != "SUCCESS":
            status = "SKIPPED_STALE_INPUT"
            results.append(("Candidate Training and Validation", status))
            print(
                "Candidate training blocked because the retraining "
                "decision stage did not receive all required fresh inputs."
            )
        else:
            try:
                decision_report = read_json(DECISION_FILE)
                if not decision_report:
                    raise FileNotFoundError(
                        "Retraining decision report was not generated."
                    )

                decision = decision_report.get("decision")
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

    print("::endgroup::", flush=True)
    print("\\n========== AEROAI PIPELINE SUMMARY ==========")
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
    print("\\nPipeline result:", final_status)
    print("Pipeline report:", PIPELINE_REPORT_FILE)
    return exit_code


if __name__ == "__main__":
    sys.exit(run_pipeline())
