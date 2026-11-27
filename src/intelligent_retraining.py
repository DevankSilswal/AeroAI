from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_FILE = Path("data/incoming/dehradun_aqi_processed.csv")
ACTIVE_MODEL = Path("best_aqi_model.pkl")
METRICS_FILE = Path("model_results.json")
CANDIDATE_MODEL = Path("models/cpcb_retraining_candidate.pkl")
REPORT_FILE = Path("data/monitoring/retraining_report.json")

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH",
    "PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI"
]

MINIMUM_ROWS = 90
IMPROVEMENT_REQUIRED = 0.05


def save_blocked_report(reason, rows=None):
    report = {
        "status": "Blocked",
        "reason": reason,
        "target": "Next-day CPCB_AQI",
        "available_rows": rows,
        "minimum_required_rows": MINIMUM_ROWS,
        "active_model_changed": False,
        "promotion_allowed": False
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\nRETRAINING BLOCKED")
    print("Reason:", reason)
    print("Active champion remains unchanged.")


def run_retraining():
    print("\n========== CPCB INTELLIGENT RETRAINING ==========")

    for path in [DATA_FILE, ACTIVE_MODEL, METRICS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    metrics = json.loads(METRICS_FILE.read_text())
    active_target = metrics.get("target", "Unknown")

    # A legacy champion cannot be evaluated on CPCB labels.
    if active_target != "Next-day CPCB_AQI":
        save_blocked_report(
            "Active champion target mismatch. Legacy estimated-AQI "
            "model cannot be compared with a CPCB_AQI candidate."
        )
        return

    df = pd.read_csv(DATA_FILE)

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

    df = df.dropna(
        subset=FEATURES + ["target_CPCB_AQI"]
    )

    rows = len(df)

    print("CPCB labelled observations:", rows)
    print("Minimum required observations:", MINIMUM_ROWS)

    if rows < MINIMUM_ROWS:
        save_blocked_report(
            "Insufficient CPCB labelled observations.",
            rows
        )
        return

    split = int(rows * 0.80)

    X_train = df[FEATURES].iloc[:split]
    y_train = df["target_CPCB_AQI"].iloc[:split]
    X_test = df[FEATURES].iloc[split:]
    y_test = df["target_CPCB_AQI"].iloc[split:]

    if len(X_test) < 10:
        save_blocked_report("Insufficient chronological test observations.", rows)
        return

    champion = joblib.load(ACTIVE_MODEL)

    if getattr(champion, "n_features_in_", len(FEATURES)) != len(FEATURES):
        save_blocked_report("Champion feature count mismatch.", rows)
        return

    champion_predictions = champion.predict(X_test)
    champion_mae = mean_absolute_error(y_test, champion_predictions)

    # Persistence predicts the current day's CPCB AQI for the next day.
    persistence_predictions = X_test["PM2.5_AQI"].copy()

    # Persistence baseline must use actual current-day CPCB_AQI,
    # not a pollutant sub-index. Align with the feature row's date.
    test_positions = df.index[split:]
    persistence_predictions = df.loc[
        test_positions, "CPCB_AQI"
    ].to_numpy()

    persistence_mae = mean_absolute_error(
        y_test, persistence_predictions
    )

    candidate = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    candidate.fit(X_train, y_train)
    candidate_predictions = candidate.predict(X_test)

    candidate_mae = mean_absolute_error(y_test, candidate_predictions)
    candidate_rmse = np.sqrt(
        mean_squared_error(y_test, candidate_predictions)
    )

    improvement_vs_champion = (
        (champion_mae - candidate_mae) / champion_mae
        if champion_mae > 0 else 0
    )

    beats_champion = (
        improvement_vs_champion >= IMPROVEMENT_REQUIRED
    )
    beats_persistence = candidate_mae < persistence_mae

    CANDIDATE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, CANDIDATE_MODEL)

    # No automatic promotion. Champion is preserved for separate
    # approval and verification.
    promotion_eligible = beats_champion and beats_persistence

    report = {
        "status": (
            "Candidate passed validation - manual promotion required"
            if promotion_eligible
            else "Candidate rejected"
        ),
        "target": "Next-day CPCB_AQI",
        "available_rows": rows,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "champion_test_MAE": round(float(champion_mae), 3),
        "candidate_test_MAE": round(float(candidate_mae), 3),
        "candidate_test_RMSE": round(float(candidate_rmse), 3),
        "persistence_baseline_MAE": round(float(persistence_mae), 3),
        "improvement_vs_champion_percent": round(
            float(improvement_vs_champion * 100), 2
        ),
        "required_improvement_percent": IMPROVEMENT_REQUIRED * 100,
        "beats_champion": bool(beats_champion),
        "beats_persistence": bool(beats_persistence),
        "promotion_eligible": bool(promotion_eligible),
        "active_model_changed": False,
        "candidate_model": str(CANDIDATE_MODEL),
        "note": (
            "Candidate is never promoted automatically. Manual review "
            "and explicit champion metadata update are required."
        )
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== CPCB MODEL COMPARISON ==========")
    print("Champion MAE:", round(champion_mae, 3))
    print("Candidate MAE:", round(candidate_mae, 3))
    print("Persistence MAE:", round(persistence_mae, 3))
    print("Beats champion:", beats_champion)
    print("Beats persistence:", beats_persistence)
    print("Promotion eligible:", promotion_eligible)
    print("Active model changed: NO")
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    run_retraining()
