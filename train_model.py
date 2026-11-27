import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.aqi_calculator import calculate_aqi


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data/raw/dehradun_cleaned.csv"
CANDIDATE_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data/processed"

CANDIDATE_MODEL = CANDIDATE_DIR / "cpcb_candidate_model.pkl"
CANDIDATE_RESULTS = CANDIDATE_DIR / "cpcb_candidate_results.json"
TRAINING_DATASET = PROCESSED_DIR / "dehradun_cpcb_training_dataset.csv"

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH",
    "PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI"
]

TARGET = "CPCB_AQI"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AeroAI-AQI-Model-Training")


# ==========================================
# LOAD AND VALIDATE HISTORICAL DATA
# ==========================================

print("\n========== CPCB-BASED MODEL TRAINING ==========")

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

df = pd.read_csv(DATA_FILE)
df["period_start"] = pd.to_datetime(df["period_start"])

df = (
    df.sort_values("period_start")
    .drop_duplicates(subset=["period_start"], keep="last")
    .reset_index(drop=True)
)

required = [
    "period_start", "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Ozone", "AT", "RH"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[required[1:]].isna().any().any():
    raise ValueError(
        "Historical source has missing pollutant/feature values. "
        "Resolve data quality before training."
    )

print("Historical rows:", len(df))
print("Date range:", df["period_start"].min(), "to", df["period_start"].max())


# ==========================================
# SHARED CPCB CALCULATOR
# ==========================================

print("\nCalculating AQI using shared CPCB calculator...")

df = calculate_aqi(df.copy(), mode="daily")

if TARGET not in df.columns:
    raise ValueError(
        f"Shared calculator did not generate {TARGET}. "
        "Check src/aqi_calculator.py."
    )

for feature in FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Calculator did not generate feature: {feature}")

print("\nAQI status counts:")
print(df["AQI_status"].value_counts(dropna=False).to_string())

print("\nSub-index availability:")
for pollutant in ["PM2.5", "PM10", "NO2", "SO2"]:
    col = pollutant + "_AQI"
    print(f"{col}: {df[col].notna().sum()} valid rows")

print("\nCPCB AQI availability:", df[TARGET].notna().sum())


# ==========================================
# NEXT-DAY FORECAST DATASET
# ==========================================

df["target_AQI"] = df[TARGET].shift(-1)

data = df.dropna(subset=FEATURES + ["target_AQI"]).copy()

if len(data) < 50:
    raise ValueError(
        f"Only {len(data)} labelled rows available. "
        "At least 50 are required for this training experiment."
    )

X = data[FEATURES]
y = data["target_AQI"]

split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Invalid chronological train/test split.")

print("\n========== DATASET SPLIT ==========")
print("Usable rows:", len(data))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training period:", data["period_start"].iloc[0], "to",
      data["period_start"].iloc[split - 1])
print("Testing period:", data["period_start"].iloc[split], "to",
      data["period_start"].iloc[-1])


# ==========================================
# BASELINE: PERSISTENCE FORECAST
# ==========================================

baseline_predictions = data[TARGET].iloc[split:].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\nPersistence baseline MAE:", round(float(baseline_mae), 3))
print("Persistence baseline RMSE:", round(float(baseline_rmse), 3))


# ==========================================
# MODEL DEFINITIONS
# ==========================================

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.04,
        objective="reg:squarederror",
        random_state=42
    )
}

results = {}
trained = {}


# ==========================================
# TRAIN AND TRACK WITH MLFLOW
# ==========================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    with mlflow.start_run(run_name=f"CPCB-Candidate-{name}"):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results[name] = {
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "R2": round(float(r2), 3)
        }

        trained[name] = model

        mlflow.log_params({
            "model_name": name,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(FEATURES),
            "split_method": "chronological_80_20",
            "target": "Next-day CPCB-breakpoint daily AQI estimate",
            "aqi_calculator": "src.aqi_calculator.calculate_aqi",
            "aqi_mode": "daily",
            "coverage_verified": False
        })

        mlflow.log_params({
            key: str(value)
            for key, value in model.get_params().items()
        })

        mlflow.log_metrics({
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "persistence_baseline_MAE": float(baseline_mae)
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle"
        )

        print(name)
        print("MAE:", round(float(mae), 3))
        print("RMSE:", round(float(rmse), 3))
        print("R2:", round(float(r2), 3))


# ==========================================
# SELECT CANDIDATE
# ==========================================

best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = trained[best_name]

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, CANDIDATE_MODEL)

df.to_csv(TRAINING_DATASET, index=False)

candidate_results = {
    "candidate_model": best_name,
    "model_type": "CPCB-based candidate - not promoted",
    "results": results,
    "persistence_baseline": {
        "MAE": round(float(baseline_mae), 3),
        "RMSE": round(float(baseline_rmse), 3)
    },
    "features": FEATURES,
    "target": "Next-day CPCB_AQI",
    "target_column": TARGET,
    "aqi_method": "Shared CPCB breakpoint calculator",
    "aqi_mode": "daily",
    "coverage_status": (
        "UNVERIFIED: historical data contains daily averages; "
        "16-hour pollutant coverage cannot be established."
    ),
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "training_start": str(data["period_start"].iloc[0]),
    "training_end": str(data["period_start"].iloc[split - 1]),
    "testing_start": str(data["period_start"].iloc[split]),
    "testing_end": str(data["period_start"].iloc[-1]),
    "candidate_created_at": datetime.now().astimezone().isoformat(),
    "candidate_model_path": str(CANDIDATE_MODEL),
    "promoted": False
}

with open(CANDIDATE_RESULTS, "w") as f:
    json.dump(candidate_results, f, indent=4)


# ==========================================
# FINAL SUMMARY
# ==========================================

with mlflow.start_run(run_name="AeroAI-CPCB-Candidate-Summary"):

    mlflow.set_tag("project", "AeroAI")
    mlflow.set_tag("candidate_model", best_name)
    mlflow.set_tag("promoted", "false")

    mlflow.log_param("selection_metric", "lowest_MAE")
    mlflow.log_param("target", "Next-day CPCB_AQI")
    mlflow.log_param("coverage_verified", False)

    mlflow.log_metrics({
        "candidate_MAE": results[best_name]["MAE"],
        "candidate_RMSE": results[best_name]["RMSE"],
        "candidate_R2": results[best_name]["R2"],
        "baseline_MAE": float(baseline_mae)
    })

    mlflow.log_artifact(str(CANDIDATE_RESULTS))
    mlflow.log_artifact(str(CANDIDATE_MODEL))


print("\n========== CPCB CANDIDATE RESULT ==========")
print("Candidate model:", best_name)
print("Candidate metrics:", results[best_name])
print("Persistence baseline MAE:", round(float(baseline_mae), 3))
print("Candidate model:", CANDIDATE_MODEL)
print("Candidate metadata:", CANDIDATE_RESULTS)
print("Training dataset:", TRAINING_DATASET)
print("Active champion: NOT MODIFIED")
print("Automatic promotion: DISABLED")
print("\nCPCB candidate training completed.")
