import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aqi_calculator
import drift_detection
import preprocessing
import intelligent_retraining


class TestAQICalculator(unittest.TestCase):

    def test_pm25_breakpoint(self):
        self.assertEqual(aqi_calculator.sub_index(30, "PM2.5"), 50)

    def test_negative_value_returns_nan(self):
        result = aqi_calculator.sub_index(-5, "PM2.5")
        self.assertTrue(np.isnan(result))

    def test_value_above_maximum_returns_500(self):
        self.assertEqual(
            aqi_calculator.sub_index(600, "PM2.5"), 500
        )

    def test_missing_value_returns_nan(self):
        result = aqi_calculator.sub_index(np.nan, "PM2.5")
        self.assertTrue(np.isnan(result))

    def test_estimated_aqi_is_maximum_subindex(self):
        data = pd.DataFrame([{
            "PM2.5": 30,
            "PM10": 50,
            "NO2": 40,
            "SO2": 40,
            "AT": 20,
            "RH": 50
        }])

        result = aqi_calculator.calculate_aqi(data)

        self.assertEqual(result.loc[0, "estimated_AQI"], 50)
        self.assertIn("PM2.5_AQI", result.columns)
        self.assertIn("SO2_AQI", result.columns)


class TestDriftDetection(unittest.TestCase):

    def test_identical_distributions_have_low_psi(self):
        reference = np.arange(100)
        current = np.arange(100)

        psi = drift_detection.calculate_psi(reference, current)

        self.assertAlmostEqual(psi, 0.0, places=5)

    def test_shifted_distribution_has_positive_psi(self):
        reference = np.arange(100)
        current = np.arange(100) + 1000

        psi = drift_detection.calculate_psi(reference, current)

        self.assertGreater(psi, 0.20)

    def test_insufficient_reference_data_returns_nan(self):
        reference = np.arange(10)
        current = np.arange(10)

        psi = drift_detection.calculate_psi(reference, current)

        self.assertTrue(np.isnan(psi))

    def test_insufficient_current_data_returns_nan(self):
        reference = np.arange(100)
        current = np.arange(4)

        psi = drift_detection.calculate_psi(reference, current)

        self.assertTrue(np.isnan(psi))


class TestPreprocessing(unittest.TestCase):

    def test_duplicate_average_and_incomplete_date_removal(self):
        pollutants = preprocessing.REQUIRED_POLLUTANTS

        rows = []

        # Complete date with duplicate PM2.5 readings
        for pollutant in pollutants:
            value = 10.0
            rows.append({
                "station_id": "test_station",
                "parameter_name": pollutant,
                "period_start": "2026-01-01",
                "mean": value,
                "unit": preprocessing.EXPECTED_UNITS[pollutant]
            })

        rows.append({
            "station_id": "test_station",
            "parameter_name": "PM2.5",
            "period_start": "2026-01-01",
            "mean": 30.0,
            "unit": preprocessing.EXPECTED_UNITS["PM2.5"]
        })

        # Incomplete date: only PM2.5 is present
        rows.append({
            "station_id": "test_station",
            "parameter_name": "PM2.5",
            "period_start": "2026-01-02",
            "mean": 20.0,
            "unit": preprocessing.EXPECTED_UNITS["PM2.5"]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.csv"
            output_file = Path(temp_dir) / "output.csv"

            pd.DataFrame(rows).to_csv(input_file, index=False)

            with patch.object(preprocessing, "INPUT_FILE", input_file), \
                 patch.object(preprocessing, "OUTPUT_FILE", output_file):

                preprocessing.preprocess_data()

            result = pd.read_csv(output_file)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "period_start"], "2026-01-01")
            self.assertAlmostEqual(result.loc[0, "PM2.5"], 20.0)

            for pollutant in pollutants:
                self.assertFalse(result[pollutant].isna().any())


class TestIntelligentRetraining(unittest.TestCase):

    def test_retraining_blocked_when_data_is_insufficient(self):
        features = intelligent_retraining.FEATURES
        rows = []

        for i in range(5):
            row = {
                "period_start": f"2026-01-0{i + 1}",
                "CPCB_AQI": 50 + i
            }
            for feature in features:
                row[feature] = 10.0 + i
            rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            data_file = temp / "processed.csv"
            active_model = temp / "active_model.pkl"
            report_file = temp / "retraining_report.json"
            metrics_file = temp / "model_results.json"

            pd.DataFrame(rows).to_csv(data_file, index=False)
            active_model.write_text("original-model-placeholder")
            metrics_file.write_text(json.dumps({
                "target": "Next-day CPCB_AQI"
            }))

            with patch.object(intelligent_retraining, "DATA_FILE", data_file), \
                 patch.object(intelligent_retraining, "ACTIVE_MODEL", active_model), \
                 patch.object(intelligent_retraining, "METRICS_FILE", metrics_file), \
                 patch.object(intelligent_retraining, "REPORT_FILE", report_file):
                intelligent_retraining.run_retraining()

            report = json.loads(report_file.read_text())
            self.assertEqual(report["status"], "Blocked")
            self.assertIn(
                "Insufficient CPCB labelled observations",
                report["reason"]
            )
            self.assertEqual(report["available_rows"], 4)
            self.assertFalse(report["active_model_changed"])
            self.assertEqual(
                active_model.read_text(),
                "original-model-placeholder"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
