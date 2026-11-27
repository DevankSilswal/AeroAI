import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prediction
import model_monitoring
from aqi_calculator import FEATURES


class FakeModel:

    def predict(self, X):
        assert list(X.columns) == FEATURES
        return np.array([650.0])


class TestPrediction(unittest.TestCase):

    def test_prediction_clipping_and_history_replacement(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            model_file = temp / "model.pkl"
            metrics_file = temp / "metrics.json"
            data_file = temp / "processed.csv"
            history_file = temp / "history.csv"

            model_file.touch()

            metrics = {
                "features": FEATURES,
                "best_model": "XGBoost",
                "training_rows": 292,
                "results": {
                    "XGBoost": {"MAE": 39.588}
                }
            }

            metrics_file.write_text(json.dumps(metrics))

            data = []

            for date in ["2026-01-01", "2026-01-02"]:
                row = {"period_start": date}

                for feature in FEATURES:
                    row[feature] = 10.0

                data.append(row)

            pd.DataFrame(data).to_csv(data_file, index=False)

            # Existing forecast for the same target date.
            history_file.write_text(
                "source_date,forecast_date,predicted_estimated_AQI\n"
                "2025-12-31,2026-01-03,100\n"
                "2025-12-30,2025-12-31,90\n"
            )

            with patch.object(prediction, "MODEL_FILE", model_file), \
                 patch.object(prediction, "METRICS_FILE", metrics_file), \
                 patch.object(prediction, "DATA_FILE", data_file), \
                 patch.object(prediction, "HISTORY_FILE", history_file), \
                 patch.object(prediction, "MAX_DATA_AGE_HOURS", 100000), \
                 patch.object(prediction.joblib, "load", return_value=FakeModel()):

                prediction.generate_prediction()

            result = pd.read_csv(history_file)

            matching = result[
                result["forecast_date"] == "2026-01-03"
            ]

            self.assertEqual(len(matching), 1)
            self.assertEqual(
                matching.iloc[0]["predicted_estimated_AQI"], 500.0
            )
            self.assertEqual(
                matching.iloc[0]["source_date"], "2026-01-02"
            )
            self.assertEqual(len(result), 2)

    def test_feature_mismatch_is_rejected(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            model_file = temp / "model.pkl"
            metrics_file = temp / "metrics.json"
            data_file = temp / "processed.csv"

            model_file.touch()
            data_file.touch()

            metrics_file.write_text(json.dumps({
                "features": ["wrong_feature"],
                "best_model": "XGBoost"
            }))

            with patch.object(prediction, "MODEL_FILE", model_file), \
                 patch.object(prediction, "METRICS_FILE", metrics_file), \
                 patch.object(prediction, "DATA_FILE", data_file), \
                 patch.object(prediction.joblib, "load", return_value=FakeModel()):

                with self.assertRaises(ValueError):
                    prediction.generate_prediction()


class TestModelMonitoring(unittest.TestCase):

    def test_evaluated_and_pending_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            history_file = temp / "history.csv"
            actual_file = temp / "actual.csv"
            output_file = temp / "evaluation.csv"
            summary_file = temp / "summary.json"
            metrics_file = temp / "model_results.json"
            metrics_file.write_text(json.dumps({
                "target": "Next-day CPCB_AQI"
            }))

            predictions = pd.DataFrame([
                {
                    "forecast_date": "2026-01-01",
                    "predicted_estimated_AQI": 100,
                    "prediction_target": "Next-day CPCB_AQI",
                    "forecast_type": "CPCB_AQI_FORECAST"
                },
                {
                    "forecast_date": "2026-01-02",
                    "predicted_estimated_AQI": 80,
                    "prediction_target": "Next-day CPCB_AQI",
                    "forecast_type": "CPCB_AQI_FORECAST"
                }
            ])

            actuals = pd.DataFrame([{
                "period_start": "2026-01-01",
                "CPCB_AQI": 120
            }])

            predictions.to_csv(history_file, index=False)
            actuals.to_csv(actual_file, index=False)

            with patch.object(model_monitoring, "HISTORY_FILE", history_file), \
                 patch.object(model_monitoring, "ACTUAL_FILE", actual_file), \
                 patch.object(model_monitoring, "METRICS_FILE", metrics_file), \
                 patch.object(model_monitoring, "OUTPUT_FILE", output_file), \
                 patch.object(model_monitoring, "SUMMARY_FILE", summary_file):
                model_monitoring.monitor_predictions()

            result = pd.read_csv(output_file)
            summary = json.loads(summary_file.read_text())

            self.assertEqual(summary["total_predictions"], 2)
            self.assertEqual(summary["evaluated_predictions"], 1)
            self.assertEqual(summary["pending_predictions"], 1)
            self.assertAlmostEqual(summary["observed_MAE"], 20.0)
            self.assertAlmostEqual(summary["observed_RMSE"], 20.0)
            self.assertEqual(
                result.loc[0, "evaluation_status"],
                "Evaluated against CPCB_AQI"
            )
            self.assertEqual(
                result.loc[1, "evaluation_status"],
                "Actual outcome pending"
            )

    def test_all_predictions_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            history_file = temp / "history.csv"
            actual_file = temp / "actual.csv"
            output_file = temp / "evaluation.csv"
            summary_file = temp / "summary.json"
            metrics_file = temp / "model_results.json"
            metrics_file.write_text(json.dumps({
                "target": "Next-day CPCB_AQI"
            }))

            pd.DataFrame([{
                "forecast_date": "2026-01-02",
                "predicted_estimated_AQI": 100,
                "prediction_target": "Next-day CPCB_AQI",
                "forecast_type": "CPCB_AQI_FORECAST"
            }]).to_csv(history_file, index=False)

            pd.DataFrame([{
                "period_start": "2026-01-01",
                "CPCB_AQI": 80
            }]).to_csv(actual_file, index=False)

            with patch.object(model_monitoring, "HISTORY_FILE", history_file), \
                 patch.object(model_monitoring, "ACTUAL_FILE", actual_file), \
                 patch.object(model_monitoring, "METRICS_FILE", metrics_file), \
                 patch.object(model_monitoring, "OUTPUT_FILE", output_file), \
                 patch.object(model_monitoring, "SUMMARY_FILE", summary_file):
                model_monitoring.monitor_predictions()

            summary = json.loads(summary_file.read_text())
            self.assertEqual(summary["evaluated_predictions"], 0)
            self.assertEqual(summary["pending_predictions"], 1)
            self.assertIsNone(summary["observed_MAE"])
            self.assertIsNone(summary["observed_RMSE"])
            self.assertEqual(
                summary["status"],
                "Waiting for matching CPCB actual observations"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
