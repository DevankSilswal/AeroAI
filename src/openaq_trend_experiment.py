import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/openaq_next_day_dataset.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_trend_experiment.json"

BASE_FEATURES = [
    "pm25_mean", "pm25_max",
    "pm10_mean", "pm10_max",
    "ozone_mean", "ozone_max",
    "temperature_mean", "humidity_mean",
    "day_end_aqi",
    "day_end_pm25_aqi",
    "day_end_pm10_aqi",
    "day_end_ozone_aqi",
]

TARGET = "next_day_cpcb_method_estimated_AQI"
TEST_SIZE = 12
N_SPLITS = 4


def metrics(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def build_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(
        df["target_date"], errors="coerce"
    )

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any() or df["target_date"].isna().any():
        raise ValueError("Invalid dates in training dataset.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate source dates detected.")

    if not (
        df["target_date"] == df["date"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError("Next-day target alignment failed.")

    by_date = {
        row["date"].normalize(): row
        for _, row in df.iterrows()
    }

    engineered = []

    for _, row in df.iterrows():
        current_date = row["date"].normalize()
        previous_1 = by_date.get(current_date - pd.Timedelta(days=1))
        previous_2 = by_date.get(current_date - pd.Timedelta(days=2))

        record = row.to_dict()

        record["aqi_lag_1"] = (
            previous_1["day_end_aqi"]
            if previous_1 is not None else np.nan
        )

        record["aqi_lag_2"] = (
            previous_2["day_end_aqi"]
            if previous_2 is not None else np.nan
        )

        record["aqi_3day_mean"] = np.nan
        record["aqi_3day_trend"] = np.nan
        record["pm25_change_1d"] = np.nan
        record["pm10_change_1d"] = np.nan
        record["ozone_change_1d"] = np.nan

        if previous_1 is not None:
            record["pm25_change_1d"] = (
                row["pm25_mean"] - previous_1["pm25_mean"]
            )
            record["pm10_change_1d"] = (
                row["pm10_mean"] - previous_1["pm10_mean"]
            )
            record["ozone_change_1d"] = (
                row["ozone_mean"] - previous_1["ozone_mean"]
            )

        if previous_1 is not None and previous_2 is not None:
            history = [
                row["day_end_aqi"],
                previous_1["day_end_aqi"],
                previous_2["day_end_aqi"],
            ]

            if all(pd.notna(value) for value in history):
                record["aqi_3day_mean"] = float(np.mean(history))
                record["aqi_3day_trend"] = (
                    float(row["day_end_aqi"])
                    - float(previous_2["day_end_aqi"])
                )

        target_day = row["target_date"]

        weekday = target_day.day_of_week
        month = target_day.month

        record["target_weekday_sin"] = np.sin(
            2 * np.pi * weekday / 7
        )
        record["target_weekday_cos"] = np.cos(
            2 * np.pi * weekday / 7
        )
        record["target_month_sin"] = np.sin(
            2 * np.pi * month / 12
        )
        record["target_month_cos"] = np.cos(
            2 * np.pi * month / 12
        )

        engineered.append(record)

    result = pd.DataFrame(engineered)

    extra_features = [
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_3day_mean",
        "aqi_3day_trend",
        "pm25_change_1d",
        "pm10_change_1d",
        "ozone_change_1d",
        "target_weekday_sin",
        "target_weekday_cos",
        "target_month_sin",
        "target_month_cos",
    ]

    features = BASE_FEATURES + extra_features

    for col in features + [TARGET]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result, features


def create_models():
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


def main():
    print("\n========== OPENAQ TREND-AWARE EXPERIMENT ==========")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required = BASE_FEATURES + [
        TARGET, "date", "target_date", "coverage_pass"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df, features = build_features(df)

    usable = df.dropna(subset=features + [TARGET]).copy()
    usable = usable.sort_values("date").reset_index(drop=True)

    print("Original rows:", len(df))
    print("Rows after lag-feature validation:", len(usable))
    print("Rows removed for missing/gapped history:", len(df) - len(usable))

    if len(usable) <= N_SPLITS * TEST_SIZE:
        raise RuntimeError("Insufficient rows for walk-forward validation.")

    X = usable[features]
    y = usable[TARGET]

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE
    )

    all_actual = []
    all_baselines = {
        "Persistence": [],
        "3-day rolling mean": [],
    }
    all_predictions = {
        "Random Forest": [],
        "XGBoost": [],
    }

    folds = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(usable), start=1
    ):
        train = usable.iloc[train_idx]
        test = usable.iloc[test_idx]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx].to_numpy()

        persistence = test["day_end_aqi"].to_numpy()
        rolling_baseline = test["aqi_3day_mean"].to_numpy()

        fold_predictions = {}

        for name, model in create_models().items():
            model.fit(X_train, y_train)

            prediction = np.clip(
                model.predict(X_test), 0, 500
            )

            fold_predictions[name] = prediction
            all_predictions[name].extend(prediction.tolist())

        all_actual.extend(y_test.tolist())
        all_baselines["Persistence"].extend(persistence.tolist())
        all_baselines["3-day rolling mean"].extend(
            rolling_baseline.tolist()
        )

        fold_report = {
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": test["target_date"].min().strftime("%Y-%m-%d"),
            "test_end": test["target_date"].max().strftime("%Y-%m-%d"),
            "baselines": {
                "Persistence": metrics(y_test, persistence),
                "3-day rolling mean": metrics(y_test, rolling_baseline),
            },
            "candidates": {
                name: metrics(y_test, prediction)
                for name, prediction in fold_predictions.items()
            },
        }

        folds.append(fold_report)

        print(f"\nFold {fold}")
        print("Persistence MAE:", fold_report["baselines"]["Persistence"]["MAE"])
        print("3-day mean MAE:", fold_report["baselines"]["3-day rolling mean"]["MAE"])

        for name, score in fold_report["candidates"].items():
            print(name, "MAE:", score["MAE"])

    actual = np.asarray(all_actual)

    baseline_metrics = {
        name: metrics(actual, np.asarray(predictions))
        for name, predictions in all_baselines.items()
    }

    candidate_metrics = {
        name: metrics(actual, np.asarray(predictions))
        for name, predictions in all_predictions.items()
    }

    best_baseline_name = min(
        baseline_metrics,
        key=lambda name: baseline_metrics[name]["MAE"]
    )

    best_candidate_name = min(
        candidate_metrics,
        key=lambda name: candidate_metrics[name]["MAE"]
    )

    best_baseline_mae = baseline_metrics[best_baseline_name]["MAE"]
    best_candidate_mae = candidate_metrics[best_candidate_name]["MAE"]

    improvement = (
        (best_baseline_mae - best_candidate_mae)
        / best_baseline_mae * 100
        if best_baseline_mae > 0 else 0.0
    )

    fold_wins = sum(
        folds[i]["candidates"][best_candidate_name]["MAE"]
        < min(
            folds[i]["baselines"]["Persistence"]["MAE"],
            folds[i]["baselines"]["3-day rolling mean"]["MAE"]
        )
        for i in range(N_SPLITS)
    )

    gate = improvement >= 5 and fold_wins >= 3

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment": "Lag, trend and calendar features",
        "target": "Next-day OpenAQ CPCB-method estimated AQI",
        "official_cpcb_aqi": False,
        "rows_before_features": len(df),
        "rows_after_features": len(usable),
        "features": features,
        "validation": "Expanding-window chronological walk-forward",
        "fold_count": N_SPLITS,
        "test_rows_per_fold": TEST_SIZE,
        "baselines": baseline_metrics,
        "candidate_results": candidate_metrics,
        "best_baseline": best_baseline_name,
        "best_candidate": best_candidate_name,
        "best_baseline_MAE": best_baseline_mae,
        "best_candidate_MAE": best_candidate_mae,
        "improvement_over_best_baseline_percent": round(improvement, 2),
        "folds_won_against_strongest_baseline": fold_wins,
        "required_fold_wins": 3,
        "preliminary_gate_pass": bool(gate),
        "production_promotion": False,
        "existing_champion_unchanged": True,
        "fold_results": folds,
        "limitations": [
            "Only 48 out-of-fold predictions are evaluated.",
            "Lag features are used only when the exact prior calendar dates exist.",
            "Target is an OpenAQ-derived CPCB-method estimate, not official CPCB AQI.",
            "Calendar features do not encode Diwali or other event information.",
            "This experiment does not save or promote a model."
        ],
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== TREND EXPERIMENT SUMMARY ==========")
    print("Rows evaluated:", len(actual))
    print("Baselines:", baseline_metrics)
    print("Candidates:", candidate_metrics)
    print("Best baseline:", best_baseline_name)
    print("Best candidate:", best_candidate_name)
    print("Improvement:", round(improvement, 2), "%")
    print("Folds beating both baselines:", fold_wins, "/", N_SPLITS)
    print("Preliminary gate:", gate)
    print("Production promotion: DISABLED")
    print("Existing champion: UNCHANGED")
    print("Report saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
