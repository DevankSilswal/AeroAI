import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/processed/openaq_next_day_dataset.csv"
QUALITY_FILE = ROOT / "data/monitoring/openaq_daily_dataset_quality.json"
HOURLY_FILE = ROOT / "data/processed/openaq_hourly_aqi.csv"

MODEL_DIR = ROOT / "models"
CANDIDATE_FILE = MODEL_DIR / "openaq_demo_candidate.pkl"
CHAMPION_FILE = MODEL_DIR / "openaq_demo_champion.pkl"
REGISTRY_FILE = MODEL_DIR / "openaq_demo_registry.json"
VERSIONS_DIR = MODEL_DIR / "openaq_demo_versions"

HOLDOUT_FILE = ROOT / "data/processed/openaq_demo_holdout_predictions.csv"
HISTORY_FILE = ROOT / "data/predictions/openaq_demo_prediction_history.csv"
REPORT_FILE = ROOT / "data/monitoring/openaq_demo_workflow.json"

FEATURES = [
    "pm25_mean", "pm25_max",
    "pm10_mean", "pm10_max",
    "ozone_mean", "ozone_max",
    "temperature_mean", "humidity_mean",
    "day_end_aqi", "day_end_pm25_aqi",
    "day_end_pm10_aqi", "day_end_ozone_aqi",
]
TARGET = "next_day_cpcb_method_estimated_AQI"
MINIMUM_ROWS = 90
MINIMUM_NEW_LABELS_FOR_RETRAINING = 20
LOCAL_TZ = "Asia/Kolkata"


def now_iso():
    return datetime.now().astimezone().isoformat()


def atomic_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=4, default=str))
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_factory(name):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=2,
        )

    return XGBRegressor(
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
    )


def calculate_metrics(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def improvement_percent(baseline, candidate):
    if baseline is None or baseline <= 0:
        return 0.0
    return round((baseline - candidate) / baseline * 100, 2)


def load_training_data():
    if not DATA_FILE.exists() or not QUALITY_FILE.exists():
        raise FileNotFoundError("OpenAQ daily dataset or quality report missing.")

    quality = json.loads(QUALITY_FILE.read_text())

    if quality.get("training_data_gate_pass") is not True:
        raise RuntimeError("OpenAQ daily dataset quality gate has not passed.")

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [
        TARGET, "date", "target_date", "coverage_pass"
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")

    df["coverage_pass"] = (
        df["coverage_pass"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    if df[["date", "target_date"]].isna().any().any():
        raise ValueError("Invalid source or target dates.")

    if not (
        df["target_date"] == df["date"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError("Next-day target alignment failed.")

    if df.duplicated("date").any():
        raise ValueError("Duplicate source dates detected.")

    df = df[df["coverage_pass"]].dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    if len(df) < MINIMUM_ROWS:
        raise RuntimeError(
            f"Only {len(df)} labelled rows; {MINIMUM_ROWS} required."
        )

    return df


def initial_validation(df):
    split = int(len(df) * 0.80)

    train = df.iloc[:split]
    test = df.iloc[split:]

    if len(train) < 40 or len(test) < 20:
        raise RuntimeError("Insufficient chronological train/holdout rows.")

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    persistence = X_test["day_end_aqi"].clip(0, 500)
    baseline_metrics = calculate_metrics(y_test, persistence)

    model_results = {}
    fitted_models = {}
    model_predictions = {}

    for name in ["Random Forest", "XGBoost"]:
        model = model_factory(name)
        model.fit(X_train, y_train)

        predictions = np.clip(model.predict(X_test), 0, 500)

        fitted_models[name] = model
        model_predictions[name] = predictions
        model_results[name] = calculate_metrics(y_test, predictions)

    selected_name = min(
        model_results,
        key=lambda name: model_results[name]["MAE"]
    )

    selected_predictions = model_predictions[selected_name]
    selected_metrics = model_results[selected_name]

    holdout_improvement = improvement_percent(
        baseline_metrics["MAE"], selected_metrics["MAE"]
    )

    # Four expanding-window chronological validation folds.
    test_size = min(12, max(6, (len(df) - 50) // 4))
    initial_train_size = len(df) - 4 * test_size

    fold_results = []
    all_actual = []
    all_candidate = []
    all_persistence = []

    if initial_train_size >= 40:
        for fold in range(4):
            train_end = initial_train_size + fold * test_size
            test_end = train_end + test_size

            fold_train = df.iloc[:train_end]
            fold_test = df.iloc[train_end:test_end]

            if len(fold_test) < 2:
                continue

            fold_model = model_factory(selected_name)
            fold_model.fit(
                fold_train[FEATURES],
                fold_train[TARGET]
            )

            actual = fold_test[TARGET].to_numpy()
            candidate_prediction = np.clip(
                fold_model.predict(fold_test[FEATURES]), 0, 500
            )
            baseline_prediction = fold_test[
                "day_end_aqi"
            ].clip(0, 500).to_numpy()

            candidate_mae = float(
                mean_absolute_error(actual, candidate_prediction)
            )
            baseline_mae = float(
                mean_absolute_error(actual, baseline_prediction)
            )

            fold_results.append({
                "fold": fold + 1,
                "training_rows": int(len(fold_train)),
                "test_rows": int(len(fold_test)),
                "candidate_MAE": round(candidate_mae, 4),
                "persistence_MAE": round(baseline_mae, 4),
                "candidate_won": bool(candidate_mae < baseline_mae),
            })

            all_actual.extend(actual.tolist())
            all_candidate.extend(candidate_prediction.tolist())
            all_persistence.extend(baseline_prediction.tolist())

    if all_actual:
        walk_candidate = calculate_metrics(
            all_actual, all_candidate
        )
        walk_persistence = calculate_metrics(
            all_actual, all_persistence
        )
        walk_improvement = improvement_percent(
            walk_persistence["MAE"], walk_candidate["MAE"]
        )
        fold_wins = sum(
            item["candidate_won"] for item in fold_results
        )
    else:
        walk_candidate = {}
        walk_persistence = {}
        walk_improvement = 0.0
        fold_wins = 0

    walk_forward = {
        "folds": fold_results,
        "folds_won": int(fold_wins),
        "fold_count": len(fold_results),
        "candidate_metrics": walk_candidate,
        "persistence_metrics": walk_persistence,
        "MAE_improvement_percent": walk_improvement,
    }

    # This is deliberately a DEMO-only gate, not production approval.
    eligible = (
        len(df) >= MINIMUM_ROWS
        and len(test) >= 20
        and holdout_improvement >= 5
        and walk_improvement >= 5
        and fold_wins >= 2
        and len(fold_results) == 4
    )

    holdout = pd.DataFrame({
        "source_date": test["date"].dt.strftime("%Y-%m-%d"),
        "target_date": test["target_date"].dt.strftime("%Y-%m-%d"),
        "actual_next_day_estimated_AQI": y_test.to_numpy(),
        "persistence_prediction": persistence.to_numpy(),
        "candidate_prediction": selected_predictions,
    })

    validation = {
        "mode": "INITIAL_CHRONOLOGICAL_HOLDOUT_AND_WALK_FORWARD",
        "total_rows": int(len(df)),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "train_start": train["date"].min().strftime("%Y-%m-%d"),
        "train_end": train["date"].max().strftime("%Y-%m-%d"),
        "holdout_start": test["target_date"].min().strftime("%Y-%m-%d"),
        "holdout_end": test["target_date"].max().strftime("%Y-%m-%d"),
        "persistence_baseline": baseline_metrics,
        "candidate_results": model_results,
        "selected_candidate": selected_name,
        "selected_candidate_metrics": selected_metrics,
        "holdout_MAE_improvement_percent": holdout_improvement,
        "walk_forward": walk_forward,
        "demo_promotion_eligible": bool(eligible),
        "demo_gate": {
            "minimum_rows": MINIMUM_ROWS,
            "minimum_holdout_rows": 20,
            "required_holdout_improvement_percent": 5,
            "required_walk_forward_improvement_percent": 5,
            "required_fold_wins": 2,
            "production_approval": False,
        },
    }

    return validation, holdout, selected_name


def build_latest_complete_day_features():
    if not HOURLY_FILE.exists():
        raise FileNotFoundError(HOURLY_FILE)

    hourly = pd.read_csv(HOURLY_FILE)

    required = [
        "period_start", "PM2.5", "PM10", "Ozone", "AT", "RH",
        "CPCB_AQI", "PM2.5_AQI", "PM10_AQI", "Ozone_AQI",
    ]

    missing = [column for column in required if column not in hourly.columns]
    if missing:
        raise ValueError(f"Missing hourly forecast columns: {missing}")

    hourly["period_start"] = pd.to_datetime(
        hourly["period_start"], errors="coerce", utc=True
    )
    hourly = hourly.dropna(subset=["period_start"]).copy()

    for column in required:
        if column != "period_start":
            hourly[column] = pd.to_numeric(
                hourly[column], errors="coerce"
            )

    hourly["local_date"] = (
        hourly["period_start"]
        .dt.tz_convert(LOCAL_TZ)
        .dt.strftime("%Y-%m-%d")
    )

    dates = sorted(hourly["local_date"].dropna().unique())

    if len(dates) < 2:
        raise RuntimeError("Not enough dates to build a one-day-ahead forecast.")

    # The latest date is treated as partial and excluded.
    latest_observed_date = dates[-1]
    source_date = dates[-2]

    group = hourly[
        hourly["local_date"] == source_date
    ].sort_values("period_start").copy()

    counts = {
        "pm25_hours": int(group["PM2.5"].notna().sum()),
        "pm10_hours": int(group["PM10"].notna().sum()),
        "ozone_hours": int(group["Ozone"].notna().sum()),
        "temperature_hours": int(group["AT"].notna().sum()),
        "humidity_hours": int(group["RH"].notna().sum()),
        "valid_aqi_hours": int(group["CPCB_AQI"].notna().sum()),
    }

    coverage_pass = (
        counts["pm25_hours"] >= 16
        and counts["pm10_hours"] >= 16
        and counts["ozone_hours"] >= 8
        and counts["temperature_hours"] >= 12
        and counts["humidity_hours"] >= 12
        and counts["valid_aqi_hours"] >= 16
    )

    if not coverage_pass:
        raise RuntimeError(
            f"Latest complete day {source_date} failed hourly coverage: {counts}"
        )

    valid_aqi = group.dropna(subset=["CPCB_AQI"])
    last_valid = valid_aqi.iloc[-1]

    row = {
        "pm25_mean": group["PM2.5"].mean(),
        "pm25_max": group["PM2.5"].max(),
        "pm10_mean": group["PM10"].mean(),
        "pm10_max": group["PM10"].max(),
        "ozone_mean": group["Ozone"].mean(),
        "ozone_max": group["Ozone"].max(),
        "temperature_mean": group["AT"].mean(),
        "humidity_mean": group["RH"].mean(),
        "day_end_aqi": last_valid["CPCB_AQI"],
        "day_end_pm25_aqi": last_valid["PM2.5_AQI"],
        "day_end_pm10_aqi": last_valid["PM10_AQI"],
        "day_end_ozone_aqi": last_valid["Ozone_AQI"],
    }

    features = pd.DataFrame([[row[column] for column in FEATURES]],
                            columns=FEATURES)

    if features.isna().any(axis=None):
        raise RuntimeError("Latest complete-day forecast features contain missing values.")

    source = pd.Timestamp(source_date)
    target = source + pd.Timedelta(days=1)

    return (
        features,
        source_date,
        target.strftime("%Y-%m-%d"),
        latest_observed_date,
        counts,
    )


def aqi_category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def main():
    print("\n========== OPENAQ DEMO PREDICTION & RETRAINING ==========")

    df = load_training_data()
    fingerprint = file_hash(DATA_FILE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    registry = (
        json.loads(REGISTRY_FILE.read_text())
        if REGISTRY_FILE.exists() else {}
    )

    champion_exists = CHAMPION_FILE.exists()
    prior_seen_hash = registry.get("last_seen_data_hash")
    new_data = fingerprint != prior_seen_hash

    validation = registry.get("validation", {})
    selected_name = registry.get("model_family")
    action = "NO_NEW_LABELLED_DATA"
    reason = "Dataset fingerprint is unchanged; existing demo champion retained."
    candidate_model = None
    holdout = (
        pd.read_csv(HOLDOUT_FILE)
        if HOLDOUT_FILE.exists() else pd.DataFrame()
    )

    if not registry or not champion_exists:
        validation, holdout, selected_name = initial_validation(df)
        holdout.to_csv(HOLDOUT_FILE, index=False)

        selected_eligible = validation["demo_promotion_eligible"]

        candidate_model = model_factory(selected_name)
        candidate_model.fit(df[FEATURES], df[TARGET])

        joblib.dump(
            candidate_model,
            CANDIDATE_FILE.with_suffix(".pkl.tmp")
        )
        CANDIDATE_FILE.with_suffix(".pkl.tmp").replace(CANDIDATE_FILE)

        if selected_eligible:
            action = "PROMOTED_DEMO_ONLY"
            reason = (
                "Candidate passed the demo chronological holdout and "
                "walk-forward gates. Promotion is restricted to the "
                "separate OpenAQ demo registry."
            )

            version = datetime.now().strftime("demo_%Y%m%d_%H%M%S")

            if CHAMPION_FILE.exists():
                previous_version = registry.get("model_version", "previous")
                shutil.copy2(
                    CHAMPION_FILE,
                    VERSIONS_DIR / f"openaq_demo_champion_{previous_version}.pkl"
                )

            temporary = CHAMPION_FILE.with_suffix(".pkl.tmp")
            joblib.dump(candidate_model, temporary)
            temporary.replace(CHAMPION_FILE)

            registry = {
                "environment": "DEMO_ONLY",
                "model_version": version,
                "model_family": selected_name,
                "model_file": str(CHAMPION_FILE),
                "target": "Next-day OpenAQ CPCB-method estimated AQI",
                "official_cpcb_aqi": False,
                "production_promotion": False,
                "data_fingerprint": fingerprint,
                "last_seen_data_hash": fingerprint,
                "trained_at": now_iso(),
                "trained_through_target_date": df["target_date"].max().strftime("%Y-%m-%d"),
                "trained_rows": int(len(df)),
                "validation": validation,
            }
            atomic_json(registry, REGISTRY_FILE)

        else:
            action = "CANDIDATE_REJECTED"
            reason = (
                "Candidate did not pass all demo validation gates. "
                "The candidate is retained separately; no demo champion "
                "was replaced."
            )
            registry.update({
                "environment": "DEMO_ONLY",
                "last_seen_data_hash": fingerprint,
                "validation": validation,
            })
            atomic_json(registry, REGISTRY_FILE)

    elif new_data:
        cutoff = pd.to_datetime(
            registry.get("trained_through_target_date"), errors="coerce"
        )

        if pd.isna(cutoff):
            new_rows = pd.DataFrame()
        else:
            new_rows = df[df["target_date"] > cutoff].copy()

        if len(new_rows) < MINIMUM_NEW_LABELS_FOR_RETRAINING:
            action = "WAITING_FOR_NEW_LABELLED_DATA"
            reason = (
                f"{len(new_rows)} new labelled rows since the previous "
                f"demo champion; {MINIMUM_NEW_LABELS_FOR_RETRAINING} "
                "new rows are required for a fresh temporal evaluation."
            )
            registry["last_seen_data_hash"] = fingerprint
            atomic_json(registry, REGISTRY_FILE)

        else:
            old_rows = df[df["target_date"] <= cutoff].copy()

            if len(old_rows) < 60:
                raise RuntimeError("Insufficient historical rows for a new candidate.")

            X_train = old_rows[FEATURES]
            y_train = old_rows[TARGET]
            X_test = new_rows[FEATURES]
            y_test = new_rows[TARGET]

            incumbent = joblib.load(CHAMPION_FILE)
            incumbent_prediction = np.clip(
                incumbent.predict(X_test), 0, 500
            )
            persistence = new_rows["day_end_aqi"].clip(0, 500).to_numpy()

            model_results = {}
            predictions = {}
            models = {}

            for name in ["Random Forest", "XGBoost"]:
                model = model_factory(name)
                model.fit(X_train, y_train)
                pred = np.clip(model.predict(X_test), 0, 500)

                models[name] = model
                predictions[name] = pred
                model_results[name] = calculate_metrics(y_test, pred)

            selected_name = min(
                model_results,
                key=lambda name: model_results[name]["MAE"]
            )

            candidate_prediction = predictions[selected_name]
            candidate_metrics = model_results[selected_name]
            incumbent_metrics = calculate_metrics(
                y_test, incumbent_prediction
            )
            baseline_metrics = calculate_metrics(y_test, persistence)

            vs_incumbent = improvement_percent(
                incumbent_metrics["MAE"], candidate_metrics["MAE"]
            )
            vs_persistence = improvement_percent(
                baseline_metrics["MAE"], candidate_metrics["MAE"]
            )

            eligible = vs_incumbent >= 5 and vs_persistence >= 5

            holdout = pd.DataFrame({
                "source_date": new_rows["date"].dt.strftime("%Y-%m-%d"),
                "target_date": new_rows["target_date"].dt.strftime("%Y-%m-%d"),
                "actual_next_day_estimated_AQI": y_test.to_numpy(),
                "persistence_prediction": persistence,
                "incumbent_prediction": incumbent_prediction,
                "candidate_prediction": candidate_prediction,
            })
            holdout.to_csv(HOLDOUT_FILE, index=False)

            validation = {
                "mode": "NEW_LABELS_TEMPORAL_CHAMPION_COMPARISON",
                "training_rows": int(len(old_rows)),
                "new_holdout_rows": int(len(new_rows)),
                "persistence_baseline": baseline_metrics,
                "incumbent_demo_champion": incumbent_metrics,
                "candidate_results": model_results,
                "selected_candidate": selected_name,
                "selected_candidate_metrics": candidate_metrics,
                "improvement_vs_incumbent_percent": vs_incumbent,
                "improvement_vs_persistence_percent": vs_persistence,
                "demo_promotion_eligible": bool(eligible),
                "production_approval": False,
            }

            candidate_model = model_factory(selected_name)
            candidate_model.fit(df[FEATURES], df[TARGET])

            if eligible:
                previous_version = registry.get("model_version", "previous")
                shutil.copy2(
                    CHAMPION_FILE,
                    VERSIONS_DIR / f"openaq_demo_champion_{previous_version}.pkl"
                )

                version = datetime.now().strftime("demo_%Y%m%d_%H%M%S")
                temporary = CHAMPION_FILE.with_suffix(".pkl.tmp")
                joblib.dump(candidate_model, temporary)
                temporary.replace(CHAMPION_FILE)

                action = "PROMOTED_DEMO_ONLY"
                reason = (
                    "Candidate beat both the incumbent demo champion and "
                    "persistence on newly labelled temporal holdout rows."
                )

                registry.update({
                    "model_version": version,
                    "model_family": selected_name,
                    "data_fingerprint": fingerprint,
                    "last_seen_data_hash": fingerprint,
                    "trained_at": now_iso(),
                    "trained_through_target_date": df["target_date"].max().strftime("%Y-%m-%d"),
                    "trained_rows": int(len(df)),
                    "validation": validation,
                })
                atomic_json(registry, REGISTRY_FILE)

            else:
                action = "CANDIDATE_REJECTED"
                reason = (
                    "Candidate failed the comparison against the incumbent "
                    "demo champion and/or persistence baseline. Previous "
                    "demo champion retained."
                )
                registry["last_seen_data_hash"] = fingerprint
                registry["validation"] = validation
                atomic_json(registry, REGISTRY_FILE)

    else:
        # No new labelled data; do not retrain the same dataset repeatedly.
        action = "NO_NEW_LABELLED_DATA"
        if not validation and REGISTRY_FILE.exists():
            validation = json.loads(REGISTRY_FILE.read_text()).get("validation", {})

    # Choose the protected demo champion, or an explicitly unpromoted candidate.
    if CHAMPION_FILE.exists():
        active_model = joblib.load(CHAMPION_FILE)
        active_family = registry.get("model_family", selected_name or "Unknown")
        active_version = registry.get("model_version", "Unknown")
        active_status = "DEMO_CHAMPION"
    elif CANDIDATE_FILE.exists():
        active_model = joblib.load(CANDIDATE_FILE)
        active_family = selected_name or "Experimental candidate"
        active_version = "UNPROMOTED"
        active_status = "EXPERIMENTAL_CANDIDATE_NOT_PROMOTED"
    else:
        raise RuntimeError("No demo model is available for forecast generation.")

    # Build a one-day-ahead forecast from the latest complete local day.
    latest_features, source_date, target_date, partial_date, counts = (
        build_latest_complete_day_features()
    )

    forecast_value = float(
        np.clip(active_model.predict(latest_features)[0], 0, 500)
    )

    forecast = {
        "source_date": source_date,
        "target_date": target_date,
        "forecast_AQI": round(forecast_value, 2),
        "category": aqi_category(forecast_value),
        "model_family": active_family,
        "model_version": active_version,
        "model_status": active_status,
        "horizon": "One day ahead from the latest complete local calendar day",
        "partial_latest_source_date_excluded": partial_date,
        "source": "OpenAQ API v3",
        "official_cpcb_aqi": False,
        "production_forecast": False,
        "daily_coverage": counts,
    }

    history_row = {
        **forecast,
        "generated_at": now_iso(),
        "forecast_status": "DEMO_EXPERIMENTAL",
    }

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
    else:
        history = pd.DataFrame()

    if not history.empty and "target_date" in history.columns:
        history = history[
            history["target_date"].astype(str) != target_date
        ]

    history = pd.concat(
        [history, pd.DataFrame([history_row])],
        ignore_index=True
    )
    history.to_csv(HISTORY_FILE, index=False)

    # Exercise the rejection/rollback guard with a deliberately poor test challenger.
    rollback = {
        "status": "NOT_RUN",
        "note": "No holdout predictions available for rollback verification.",
    }

    if (
        CHAMPION_FILE.exists()
        and not holdout.empty
        and "actual_next_day_estimated_AQI" in holdout.columns
    ):
        actual = pd.to_numeric(
            holdout["actual_next_day_estimated_AQI"], errors="coerce"
        )
        reference_prediction = pd.to_numeric(
            holdout["candidate_prediction"], errors="coerce"
        )
        valid = actual.notna() & reference_prediction.notna()

        if valid.any():
            active_mae = float(mean_absolute_error(
                actual[valid], reference_prediction[valid]
            ))

            degraded_prediction = np.full(
                int(valid.sum()), 500.0
            )
            degraded_mae = float(mean_absolute_error(
                actual[valid], degraded_prediction
            ))

            champion_hash_before = file_hash(CHAMPION_FILE)
            degraded_challenger_rejected = degraded_mae > active_mae
            champion_hash_after = file_hash(CHAMPION_FILE)

            rollback = {
                "status": (
                    "PASS"
                    if degraded_challenger_rejected
                    and champion_hash_before == champion_hash_after
                    else "CHECK_REQUIRED"
                ),
                "test_type": "Deliberately degraded constant-500 challenger",
                "reference_holdout_MAE": round(active_mae, 4),
                "degraded_challenger_MAE": round(degraded_mae, 4),
                "degraded_challenger_rejected": bool(
                    degraded_challenger_rejected
                ),
                "demo_champion_unchanged": (
                    champion_hash_before == champion_hash_after
                ),
                "production_champion_touched": False,
            }

    report = {
        "workflow_status": "COMPLETED",
        "generated_at": now_iso(),
        "environment": "DEMO_ONLY",
        "source": "OpenAQ API v3",
        "target": "Next-day OpenAQ CPCB-method estimated AQI",
        "official_cpcb_aqi": False,
        "production_model_changed": False,
        "retraining_action": action,
        "retraining_reason": reason,
        "new_labelled_data_detected": bool(new_data),
        "minimum_rows": MINIMUM_ROWS,
        "minimum_new_labels_for_future_retraining": MINIMUM_NEW_LABELS_FOR_RETRAINING,
        "validation": validation,
        "forecast": forecast,
        "demo_registry": registry,
        "rollback_verification": rollback,
        "holdout_predictions_file": str(HOLDOUT_FILE),
        "prediction_history_file": str(HISTORY_FILE),
        "important_note": (
            "This is an experimental OpenAQ-based academic demonstration. "
            "It is not an official CPCB AQI forecast and does not replace "
            "the production champion model."
        ),
    }

    atomic_json(report, REPORT_FILE)

    print("\n========== OPENAQ DEMO WORKFLOW RESULT ==========")
    print("Retraining action:", action)
    print("Reason:", reason)
    print("Environment: DEMO ONLY")
    print("Selected model:", active_family)
    print("Demo model status:", active_status)
    print("Forecast source date:", source_date)
    print("Forecast target date:", target_date)
    print("Forecast AQI estimate:", round(forecast_value, 2))
    print("Forecast category:", forecast["category"])
    print("Rollback protection:", rollback.get("status"))
    print("Production champion changed: NO")
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
