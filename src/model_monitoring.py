from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
import numpy as np

HISTORY_FILE = Path("data/predictions/prediction_history.csv")
ACTUAL_FILE = Path("data/incoming/dehradun_aqi_processed.csv")
METRICS_FILE = Path("model_results.json")
OUTPUT_FILE = Path("data/monitoring/prediction_evaluation.csv")
SUMMARY_FILE = Path("data/monitoring/model_monitoring.json")
MODEL_FILE = Path("best_aqi_model.pkl")


def monitor_predictions():
    for path in [HISTORY_FILE, ACTUAL_FILE, METRICS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    predictions = pd.read_csv(HISTORY_FILE)
    actuals = pd.read_csv(ACTUAL_FILE)
    metrics = json.loads(METRICS_FILE.read_text())

    model_target = metrics.get("target", "Unknown")
    is_cpcb_model = model_target == "Next-day CPCB_AQI"

    predictions["forecast_date"] = pd.to_datetime(
        predictions["forecast_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    actuals["period_start"] = pd.to_datetime(
        actuals["period_start"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    if is_cpcb_model:
        if "CPCB_AQI" not in actuals.columns:
            raise ValueError("CPCB_AQI missing from actual observations.")

        actual_column = "actual_CPCB_AQI"
        predicted_column = (
            "predicted_CPCB_AQI"
            if "predicted_CPCB_AQI" in predictions.columns
            else "predicted_estimated_AQI"
        )

        actual_subset = actuals[
            ["period_start", "CPCB_AQI"]
        ].rename(columns={
            "period_start": "forecast_date",
            "CPCB_AQI": actual_column
        })

        evaluated = predictions.merge(
            actual_subset, on="forecast_date", how="left"
        )

        # Strict validation: both fields must explicitly identify a CPCB forecast.
        # Missing metadata is treated as a legacy/untrusted forecast.
        target_match = (
            evaluated.get(
                "prediction_target",
                pd.Series("", index=evaluated.index)
            ).fillna("").astype(str).eq("Next-day CPCB_AQI")
            &
            evaluated.get(
                "forecast_type",
                pd.Series("", index=evaluated.index)
            ).fillna("").astype(str).eq("CPCB_AQI_FORECAST")
        )

        evaluated["absolute_error"] = (
            pd.to_numeric(evaluated[predicted_column], errors="coerce")
            - pd.to_numeric(evaluated[actual_column], errors="coerce")
        ).abs()

        evaluated.loc[~target_match, "absolute_error"] = np.nan

        evaluated["evaluation_status"] = np.where(
            ~target_match,
            "Skipped - forecast target mismatch",
            np.where(
                evaluated[actual_column].notna(),
                "Evaluated against CPCB_AQI",
                "Actual outcome pending"
            )
        )

        completed = evaluated.dropna(
            subset=["absolute_error", actual_column]
        )

        evaluated_target = "CPCB_AQI"

    else:
        # Legacy model predictions must not be compared with CPCB AQI.
        evaluated = predictions.copy()
        evaluated["actual_CPCB_AQI"] = np.nan
        evaluated["absolute_error"] = np.nan
        evaluated["evaluation_status"] = (
            "Skipped - active model target is not CPCB_AQI"
        )
        completed = evaluated.iloc[0:0].copy()
        evaluated_target = None

    evaluated["evaluated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(OUTPUT_FILE, index=False)

    if len(completed):
        errors = completed["absolute_error"].astype(float)
        mae = float(errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))
        status = "CPCB prediction errors available"
    else:
        mae = None
        rmse = None
        status = (
            "Awaiting CPCB-target champion and matching forecasts"
            if not is_cpcb_model
            else "Waiting for matching CPCB actual observations"
        )

    summary = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "active_model_target": model_target,
        "evaluation_target": evaluated_target,
        "total_predictions": int(len(evaluated)),
        "evaluated_predictions": int(len(completed)),
        "pending_predictions": int(
            evaluated["evaluation_status"].eq(
                "Actual outcome pending"
            ).sum()
        ),
        "skipped_predictions": int(
            evaluated["evaluation_status"].str.startswith("Skipped").sum()
        ),
        "observed_MAE": round(mae, 3) if mae is not None else None,
        "observed_RMSE": round(rmse, 3) if rmse is not None else None,
        "status": status,
        "note": (
            "Only forecasts from a CPCB_AQI-target model are evaluated "
            "against CPCB_AQI actual observations. Legacy estimated-AQI "
            "forecasts are excluded."
        )
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=4))

    print("\n========== MODEL MONITORING ==========")
    print("Active model target:", model_target)
    print("Evaluation target:", evaluated_target)
    print("Total predictions:", len(evaluated))
    print("Evaluated predictions:", len(completed))
    print("Skipped predictions:", summary["skipped_predictions"])
    print("Observed MAE:", summary["observed_MAE"])
    print("Observed RMSE:", summary["observed_RMSE"])
    print("Status:", status)
    print("Saved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)


if __name__ == "__main__":
    monitor_predictions()
