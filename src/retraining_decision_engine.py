import json
from pathlib import Path
import pandas as pd

DRIFT_FILE = Path("data/monitoring/drift_summary.json")
MONITORING_FILE = Path("data/monitoring/model_monitoring.json")
ANOMALY_FILE = Path("data/monitoring/hourly_anomaly_events.json")
METRICS_FILE = Path("model_results.json")
TRAINING_DATA_FILE = Path("data/incoming/dehradun_aqi_processed.csv")
OUTPUT_FILE = Path("data/monitoring/retraining_decision.json")

MINIMUM_ROWS = 90
MIN_DRIFT_ROWS = 30
MIN_EVALUATED_PREDICTIONS = 20
DEGRADATION_RATIO = 1.20

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH",
    "PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI"
]


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required report missing: {path}")
    return json.loads(path.read_text())


def count_cpcb_labelled_rows():
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(TRAINING_DATA_FILE)

    df = pd.read_csv(TRAINING_DATA_FILE)

    required = FEATURES + ["period_start", "CPCB_AQI"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing CPCB training columns: {missing}")

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce"
    )
    df["CPCB_AQI"] = pd.to_numeric(
        df["CPCB_AQI"], errors="coerce"
    )

    df = df.sort_values("period_start").reset_index(drop=True)
    df["target_CPCB_AQI"] = df["CPCB_AQI"].shift(-1)

    labelled = df.dropna(
        subset=FEATURES + ["target_CPCB_AQI"]
    )

    return len(labelled)


def make_retraining_decision():
    drift = read_json(DRIFT_FILE)
    monitoring = read_json(MONITORING_FILE)
    anomalies = read_json(ANOMALY_FILE)
    metrics = read_json(METRICS_FILE)

    available_rows = count_cpcb_labelled_rows()
    current_rows = int(drift.get("current_rows", 0))

    active_target = metrics.get("target", "Unknown")
    cpcb_champion_ready = active_target == "Next-day CPCB_AQI"

    psi_values = [
        item["PSI"]
        for item in drift.get("results", [])
        if item.get("PSI") is not None
    ]

    raw_high_drift = any(value >= 0.20 for value in psi_values)
    confirmed_drift = raw_high_drift and current_rows >= MIN_DRIFT_ROWS

    monitoring_target_matches = (
        monitoring.get("evaluation_target") == "CPCB_AQI"
        and monitoring.get("active_model_target") == active_target
    )

    observed_mae = monitoring.get("observed_MAE")
    evaluated = int(monitoring.get("evaluated_predictions", 0))

    best_model = metrics.get("best_model")
    baseline_mae = (
        metrics.get("results", {})
        .get(best_model, {})
        .get("MAE")
    )

    performance_assessed = (
        cpcb_champion_ready
        and monitoring_target_matches
        and evaluated >= MIN_EVALUATED_PREDICTIONS
        and observed_mae is not None
        and baseline_mae is not None
        and float(baseline_mae) > 0
    )

    performance_degraded = (
        performance_assessed
        and float(observed_mae) >
        float(baseline_mae) * DEGRADATION_RATIO
    )

    episodes = int(anomalies.get("pollution_episodes", 0))
    enough_data = available_rows >= MINIMUM_ROWS

    # Never train a CPCB candidate against a legacy champion.
    if not cpcb_champion_ready:
        decision = "MONITOR_AND_COLLECT_DATA"
        reason = (
            "Active champion target is not CPCB_AQI. A CPCB candidate "
            "cannot be compared or promoted against this legacy model. "
            "A CPCB champion must first be established through a "
            "separate, validated model-selection process."
        )

    elif not enough_data:
        decision = "WAIT_FOR_LABELLED_DATA"
        reason = (
            f"{available_rows} CPCB next-day labelled observations "
            f"available; {MINIMUM_ROWS} required."
        )

    elif confirmed_drift or performance_degraded:
        decision = "CANDIDATE_TRAINING_ELIGIBLE"
        reason = (
            "CPCB champion, sufficient labelled data, and a confirmed "
            "monitoring trigger are available. Candidate validation "
            "against champion and persistence baseline is mandatory."
        )

    else:
        decision = "MONITOR_AND_COLLECT_DATA"
        reason = "No confirmed CPCB model degradation trigger."

    report = {
        "decision": decision,
        "reason": reason,
        "active_model": best_model,
        "active_model_target": active_target,
        "cpcb_champion_ready": cpcb_champion_ready,
        "monitoring_target_matches": monitoring_target_matches,
        "monitoring_signals": {
            "raw_high_psi": raw_high_drift,
            "daily_observations": current_rows,
            "minimum_drift_observations": MIN_DRIFT_ROWS,
            "confirmed_drift": confirmed_drift,
            "evaluated_predictions": evaluated,
            "minimum_evaluated_predictions": MIN_EVALUATED_PREDICTIONS,
            "performance_assessed": bool(performance_assessed),
            "baseline_MAE": baseline_mae if cpcb_champion_ready else None,
            "observed_MAE": observed_mae if monitoring_target_matches else None,
            "performance_degraded": bool(performance_degraded),
            "pollution_episodes": episodes,
            "anomaly_signal": episodes > 0
        },
        "training_readiness": {
            "target": "CPCB_AQI",
            "available_labelled_rows": available_rows,
            "minimum_required_rows": MINIMUM_ROWS,
            "sufficient_data": enough_data
        },
        "automatic_model_replacement": False,
        "note": (
            "Only CPCB_AQI labels count toward CPCB retraining. Legacy "
            "model performance is not compared with CPCB actuals. "
            "Pollution anomalies alone do not trigger retraining."
        )
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== AEROAI RETRAINING DECISION ==========")
    print("Decision:", decision)
    print("Reason:", reason)
    print("Active model target:", active_target)
    print("CPCB champion ready:", cpcb_champion_ready)
    print("Confirmed drift:", confirmed_drift)
    print("Performance assessed:", performance_assessed)
    print("Performance degraded:", bool(performance_degraded))
    print("Pollution episodes:", episodes)
    print("CPCB labelled rows:", available_rows, "/", MINIMUM_ROWS)
    print("Automatic model replacement: DISABLED")
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    make_retraining_decision()
