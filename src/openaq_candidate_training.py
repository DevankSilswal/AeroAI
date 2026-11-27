import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/openaq_next_day_dataset.csv"
QUALITY_FILE = ROOT / "data/monitoring/openaq_daily_dataset_quality.json"

MODEL_FILE = ROOT / "models/openaq_candidate_model.pkl"
RESULT_FILE = ROOT / "models/openaq_candidate_results.json"
PREDICTIONS_FILE = ROOT / "data/processed/openaq_candidate_holdout_predictions.csv"

FEATURES = [
    "pm25_mean",
    "pm25_max",
    "pm10_mean",
    "pm10_max",
    "ozone_mean",
    "ozone_max",
    "temperature_mean",
    "humidity_mean",
    "day_end_aqi",
    "day_end_pm25_aqi",
    "day_end_pm10_aqi",
    "day_end_ozone_aqi",
]

TARGET = "next_day_cpcb_method_estimated_AQI"
MINIMUM_ROWS = 90
TRAIN_FRACTION = 0.80


def metrics(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def main():
    print("\n========== OPENAQ CANDIDATE MODEL TRAINING ==========")

    if not DATA_FILE.exists() or not QUALITY_FILE.exists():
        raise FileNotFoundError(
            "OpenAQ daily dataset or quality report is missing."
        )

    quality = json.loads(QUALITY_FILE.read_text())

    if quality.get("training_data_gate_pass") is not True:
        raise RuntimeError("Dataset quality gate has not passed.")

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [
        TARGET, "date", "target_date", "coverage_pass"
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(
        df["target_date"], errors="coerce"
    )

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    if df[["date", "target_date"]].isna().any().any():
        raise ValueError("Invalid source or target date.")

    if not (
        df["target_date"] == df["date"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "Next-day target alignment failed. Refusing to train."
        )

    if not df["coverage_pass"].astype(bool).all():
        raise ValueError("A row failed the daily coverage gate.")

    if df.duplicated("date").any():
        raise ValueError("Duplicate source dates found.")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    if len(df) < MINIMUM_ROWS:
        raise RuntimeError(
            f"Only {len(df)} usable rows; {MINIMUM_ROWS} required."
        )

    X = df[FEATURES]
    y = df[TARGET]

    split = int(len(df) * TRAIN_FRACTION)

    if split < 2 or len(df) - split < 2:
        raise RuntimeError("Insufficient chronological train/test rows.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Persistence baseline: today's end-of-day AQI is tomorrow's estimate.
    persistence = X_test["day_end_aqi"].clip(0, 500)
    baseline_metrics = metrics(y_test, persistence)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=2,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        ),
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)

        model.fit(X_train, y_train)

        pred = np.clip(model.predict(X_test), 0, 500)

        results[name] = metrics(y_test, pred)
        predictions[name] = pred

        print("Holdout metrics:", results[name])

    best_name = min(
        results,
        key=lambda name: results[name]["MAE"]
    )

    best_model = models[best_name]
    best_prediction = predictions[best_name]

    baseline_mae = baseline_metrics["MAE"]
    candidate_mae = results[best_name]["MAE"]

    improvement = (
        (baseline_mae - candidate_mae) / baseline_mae * 100
        if baseline_mae > 0 else 0.0
    )

    preliminary_improvement = (
        candidate_mae <= baseline_mae * 0.95
    )

    # Save a completely separate experimental model.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_model = MODEL_FILE.with_suffix(".pkl.tmp")
    joblib.dump(best_model, temporary_model)
    temporary_model.replace(MODEL_FILE)

    holdout = pd.DataFrame({
        "source_date": df.iloc[split:]["date"].dt.strftime("%Y-%m-%d"),
        "target_date": df.iloc[split:]["target_date"].dt.strftime("%Y-%m-%d"),
        "actual_next_day_estimated_AQI": y_test.to_numpy(),
        "persistence_prediction": persistence.to_numpy(),
        "candidate_prediction": best_prediction,
    })

    holdout.to_csv(PREDICTIONS_FILE, index=False)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "model_family": best_name,
        "model_file": str(MODEL_FILE),
        "target": "Next-day OpenAQ CPCB-method estimated AQI",
        "target_column": TARGET,
        "features": FEATURES,
        "source": "OpenAQ API v3",
        "official_cpcb_aqi": False,
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "train_start_date": df.iloc[0]["date"].strftime("%Y-%m-%d"),
        "train_end_date": df.iloc[split - 1]["date"].strftime("%Y-%m-%d"),
        "holdout_start_date": df.iloc[split]["target_date"].strftime("%Y-%m-%d"),
        "holdout_end_date": df.iloc[-1]["target_date"].strftime("%Y-%m-%d"),
        "chronological_split": True,
        "persistence_baseline": baseline_metrics,
        "candidate_results": results,
        "selected_candidate": best_name,
        "selected_candidate_metrics": results[best_name],
        "baseline_mae_improvement_percent": round(improvement, 2),
        "preliminary_5_percent_improvement_gate": bool(
            preliminary_improvement
        ),
        "promoted": False,
        "automatic_model_replacement": False,
        "holdout_predictions_file": str(PREDICTIONS_FILE),
        "validation_status": "PRELIMINARY_CHRONOLOGICAL_HOLDOUT_ONLY",
        "next_validation_required": [
            "Rolling-origin or walk-forward evaluation",
            "Performance stability across multiple time windows",
            "Candidate comparison against persistence baseline",
            "Independent actual CPCB station AQI comparison",
            "Explicit approval before any champion promotion"
        ],
    }

    RESULT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== MODEL COMPARISON ==========")
    print("Persistence baseline:", baseline_metrics)

    for name, score in results.items():
        print(name + ":", score)

    print("\nSelected candidate:", best_name)
    print("MAE improvement over persistence:", round(improvement, 2), "%")
    print("Preliminary 5% gate:", preliminary_improvement)
    print("Automatic promotion: DISABLED")
    print("Existing champion: UNCHANGED")
    print("Experimental model saved:", MODEL_FILE)
    print("Results saved:", RESULT_FILE)


if __name__ == "__main__":
    main()
