import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

from aqi_calculator import FEATURES

MODEL_FILE = Path("best_aqi_model.pkl")
METRICS_FILE = Path("model_results.json")
DATA_FILE = Path("data/incoming/dehradun_aqi_processed.csv")
HISTORY_FILE = Path("data/predictions/prediction_history.csv")
REPORT_FILE = Path("data/monitoring/prediction_freshness.json")

MAX_DATA_AGE_HOURS = 48


def write_blocked_report(reason, latest_timestamp=None, age_hours=None):
    report = {
        "status": "BLOCKED_STALE_DATA",
        "reason": reason,
        "latest_observation": (
            latest_timestamp.isoformat()
            if latest_timestamp is not None else None
        ),
        "data_age_hours": (
            round(age_hours, 2) if age_hours is not None else None
        ),
        "max_allowed_age_hours": MAX_DATA_AGE_HOURS,
        "prediction_generated": False,
        "history_modified": False,
        "checked_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== PREDICTION FRESHNESS GATE ==========")
    print("Status: BLOCKED")
    print("Reason:", reason)
    print("Latest observation:", report["latest_observation"])
    print("Data age hours:", report["data_age_hours"])
    print("Maximum allowed age:", MAX_DATA_AGE_HOURS)
    print("Prediction generated: NO")
    print("Existing prediction history: UNCHANGED")

    return False


def generate_prediction():
    for file in [MODEL_FILE, METRICS_FILE, DATA_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Required file missing: {file}")

    df = pd.read_csv(DATA_FILE)

    if "period_start" not in df.columns:
        raise ValueError("Missing period_start timestamp column.")

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce", utc=True
    )

    df = df.dropna(subset=["period_start"]).sort_values("period_start")

    if df.empty:
        return write_blocked_report("No valid observation timestamps.")

    latest_timestamp = df["period_start"].max()
    now = pd.Timestamp.now(tz="UTC")
    age_hours = (now - latest_timestamp).total_seconds() / 3600

    if age_hours < 0:
        return write_blocked_report(
            "Latest observation timestamp is in the future.",
            latest_timestamp,
            age_hours
        )

    if age_hours > MAX_DATA_AGE_HOURS:
        return write_blocked_report(
            "Latest observation exceeds the allowed data freshness window.",
            latest_timestamp,
            age_hours
        )

    usable = df.dropna(subset=FEATURES)

    if usable.empty:
        return write_blocked_report(
            "No complete model feature row is available.",
            latest_timestamp,
            age_hours
        )

    latest = usable.iloc[-1]

    # Recheck the timestamp of the actual feature row used.
    feature_timestamp = latest["period_start"]
    feature_age_hours = (
        now - feature_timestamp
    ).total_seconds() / 3600

    if feature_age_hours < 0 or feature_age_hours > MAX_DATA_AGE_HOURS:
        return write_blocked_report(
            "Latest complete feature row is stale or future-dated.",
            feature_timestamp,
            feature_age_hours
        )

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    model_target = metrics.get("target", "Unknown legacy target")
    expected_features = metrics.get("features", [])

    if expected_features != FEATURES:
        raise ValueError(
            "Model feature configuration does not match AQI calculator."
        )

    model = joblib.load(MODEL_FILE)

    X = pd.DataFrame(
        [[latest[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = float(model.predict(X)[0])
    prediction = max(0, min(500, prediction))

    source_date = feature_timestamp.date()
    forecast_date = source_date + timedelta(days=1)

    is_cpcb_model = model_target == "Next-day CPCB_AQI"

    record = {
        "source_date": str(source_date),
        "forecast_date": str(forecast_date),
        "predicted_estimated_AQI": round(prediction, 2),
        "prediction_target": model_target,
        "forecast_type": (
            "CPCB_AQI_FORECAST"
            if is_cpcb_model
            else "LEGACY_ESTIMATED_AQI_NOT_CPCB"
        ),
        "model": metrics.get("best_model", "Unknown"),
        "model_training_rows": metrics.get("training_rows"),
        "model_MAE": metrics.get("results", {})
            .get(metrics.get("best_model", ""), {})
            .get("MAE"),
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "data_age_hours": round(feature_age_hours, 2),
        "status": "Forecast - actual outcome pending"
    }

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    new_record = pd.DataFrame([record])

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
        history = history[
            history["forecast_date"].astype(str)
            != str(forecast_date)
        ]
        history = pd.concat(
            [history, new_record],
            ignore_index=True,
            sort=False
        )
    else:
        history = new_record

    history.to_csv(HISTORY_FILE, index=False)

    freshness_report = {
        "status": "FRESH",
        "reason": "Latest complete feature row is within freshness limit.",
        "latest_observation": feature_timestamp.isoformat(),
        "data_age_hours": round(feature_age_hours, 2),
        "max_allowed_age_hours": MAX_DATA_AGE_HOURS,
        "prediction_generated": True,
        "history_modified": True,
        "checked_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(freshness_report, indent=4))

    print("\n========== NEXT-DAY AQI PREDICTION ==========")
    print("Latest observation:", source_date)
    print("Data age hours:", round(feature_age_hours, 2))
    print("Forecast date:", forecast_date)
    print("Prediction:", round(prediction, 2))
    print("Model:", record["model"])
    print("Model target:", model_target)
    print("Forecast classification:", record["forecast_type"])
    print("Historical model MAE:", record["model_MAE"])
    print("Prediction history:", HISTORY_FILE)
    print("Freshness status: FRESH")
    print("Actual outcome: Pending")

    if not is_cpcb_model:
        print(
            "\nWARNING: Active model is not trained with CPCB_AQI "
            "target. This is a legacy estimated-AQI forecast, "
            "not a CPCB AQI prediction."
        )

    return True


if __name__ == "__main__":
    try:
        generate_prediction()
    except Exception as error:
        print("Prediction failed:", error)
        raise
