import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/openaq_next_day_dataset.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_walk_forward_validation.json"
CANDIDATE_FILE = ROOT / "models/openaq_candidate_model.pkl"

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
N_SPLITS = 4
TEST_SIZE = 12


def make_models():
    return {
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


def calculate_metrics(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def main():
    print("\n========== OPENAQ WALK-FORWARD VALIDATION ==========")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [TARGET, "date", "target_date"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(
        df["target_date"], errors="coerce"
    )

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET, "date", "target_date"])

    if not (
        df["target_date"] == df["date"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError("Next-day target alignment check failed.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate source dates detected.")

    if len(df) <= N_SPLITS * TEST_SIZE:
        raise ValueError("Not enough records for configured walk-forward folds.")

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE
    )

    fold_results = []
    actual_all = []
    baseline_all = []
    predictions_all = {
        "Random Forest": [],
        "XGBoost": [],
    }

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df), start=1
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET].to_numpy()

        persistence = test["day_end_aqi"].to_numpy()
        persistence = np.clip(persistence, 0, 500)

        fold_predictions = {}

        for name, model in make_models().items():
            model.fit(X_train, y_train)

            prediction = np.clip(
                model.predict(X_test), 0, 500
            )

            fold_predictions[name] = prediction
            predictions_all[name].extend(prediction.tolist())

        actual_all.extend(y_test.tolist())
        baseline_all.extend(persistence.tolist())

        baseline_metrics = calculate_metrics(y_test, persistence)

        candidate_metrics = {
            name: calculate_metrics(y_test, prediction)
            for name, prediction in fold_predictions.items()
        }

        fold_results.append({
            "fold": fold,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_end_date": train["date"].max().strftime("%Y-%m-%d"),
            "test_start_date": test["date"].min().strftime("%Y-%m-%d"),
            "test_end_date": test["date"].max().strftime("%Y-%m-%d"),
            "persistence": baseline_metrics,
            "candidates": candidate_metrics,
        })

        print(f"\nFold {fold}")
        print("Training rows:", len(train))
        print("Test rows:", len(test))
        print("Persistence MAE:", baseline_metrics["MAE"])

        for name, score in candidate_metrics.items():
            print(f"{name} MAE:", score["MAE"])

    actual = np.asarray(actual_all)
    baseline = np.asarray(baseline_all)

    baseline_overall = calculate_metrics(actual, baseline)

    overall_candidates = {
        name: calculate_metrics(actual, np.asarray(predictions))
        for name, predictions in predictions_all.items()
    }

    best_name = min(
        overall_candidates,
        key=lambda name: overall_candidates[name]["MAE"]
    )

    best_mae = overall_candidates[best_name]["MAE"]
    baseline_mae = baseline_overall["MAE"]

    improvement = (
        (baseline_mae - best_mae) / baseline_mae * 100
        if baseline_mae > 0 else 0.0
    )

    fold_wins = sum(
        fold["candidates"][best_name]["MAE"]
        < fold["persistence"]["MAE"]
        for fold in fold_results
    )

    walk_forward_gate = (
        improvement >= 5.0
        and fold_wins >= 3
    )

    # Save the candidate only if the existing experimental artifact exists.
    # This script deliberately does not fit a final model or promote anything.
    artifact_exists = CANDIDATE_FILE.exists()

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "OpenAQ historical hourly measurements",
        "target": "Next-day OpenAQ CPCB-method estimated AQI",
        "official_cpcb_aqi": False,
        "rows_evaluated": int(len(actual)),
        "fold_count": N_SPLITS,
        "test_rows_per_fold": TEST_SIZE,
        "validation_method": "Expanding-window chronological TimeSeriesSplit",
        "persistence_baseline": baseline_overall,
        "candidate_results": overall_candidates,
        "best_candidate": best_name,
        "best_candidate_MAE": best_mae,
        "MAE_improvement_over_persistence_percent": round(
            improvement, 2
        ),
        "folds_won_against_persistence": int(fold_wins),
        "folds_required_to_pass": 3,
        "preliminary_walk_forward_gate_pass": bool(walk_forward_gate),
        "candidate_artifact_exists": artifact_exists,
        "production_promotion": False,
        "fold_results": fold_results,
        "limitations": [
            "Only 48 out-of-fold observations are evaluated.",
            "The source target is an OpenAQ-derived CPCB-method estimate, not official CPCB AQI.",
            "Further independent station comparison and temporal stability checks are required.",
            "No production champion is changed by this script."
        ]
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== WALK-FORWARD SUMMARY ==========")
    print("Rows evaluated:", len(actual))
    print("Persistence:", baseline_overall)

    for name, score in overall_candidates.items():
        print(name + ":", score)

    print("\nBest candidate:", best_name)
    print("MAE improvement:", round(improvement, 2), "%")
    print("Folds beating persistence:", fold_wins, "/", N_SPLITS)
    print("Preliminary walk-forward gate:", walk_forward_gate)
    print("Production promotion: DISABLED")
    print("Existing champion: UNCHANGED")
    print("Report saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
