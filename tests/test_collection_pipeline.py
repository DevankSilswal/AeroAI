import sys
import json
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_collection
import run_pipeline


class TestDataCollection(unittest.TestCase):

    def test_collection_and_duplicate_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "collected.csv"

            existing = pd.DataFrame([
                {
                    "station_id": "site_5594",
                    "parameter_name": "PM2.5",
                    "period_start": "2026-01-01",
                    "mean": 10
                },
                {
                    "station_id": "site_5594",
                    "parameter_name": "PM10",
                    "period_start": "2026-01-01",
                    "mean": 50
                }
            ])
            existing.to_csv(output_file, index=False)

            response = Mock()
            response.text = (
                "station_id,parameter_name,period_start,mean\n"
                "site_5594,PM2.5,2026-01-01,30\n"
                "site_5594,NO2,2026-01-02,15\n"
            )
            response.raise_for_status.return_value = None

            with patch.object(data_collection, "API_KEY", "test-token"), \
                 patch.object(data_collection, "OUTPUT_FILE", output_file), \
                 patch.object(
                     data_collection.requests,
                     "get",
                     return_value=response
                 ):
                data_collection.collect_data()

            result = pd.read_csv(output_file)

            pm25 = result[
                (result["parameter_name"] == "PM2.5") &
                (result["period_start"] == "2026-01-01")
            ]

            self.assertEqual(len(pm25), 1)
            self.assertEqual(pm25.iloc[0]["mean"], 30)
            self.assertEqual(len(result), 3)
            self.assertIn("NO2", result["parameter_name"].values)

    def test_missing_api_key_raises_error(self):
        with patch.object(data_collection, "API_KEY", None), \
             patch.object(data_collection.requests, "get") as mock_get:

            with self.assertRaisesRegex(ValueError, "AQI_API_KEY missing"):
                data_collection.collect_data()

            mock_get.assert_not_called()

    def test_invalid_api_schema_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "collected.csv"

            response = Mock()
            response.text = "wrong_column,another_column\n1,2\n"
            response.raise_for_status.return_value = None

            with patch.object(data_collection, "API_KEY", "test-token"), \
                 patch.object(data_collection, "OUTPUT_FILE", output_file), \
                 patch.object(
                     data_collection.requests,
                     "get",
                     return_value=response
                 ):
                with self.assertRaisesRegex(
                    ValueError, "Unexpected API response"
                ):
                    data_collection.collect_data()


class TestPipelineOrchestration(unittest.TestCase):

    def test_all_pipeline_stages_execute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision_file = Path(temp_dir) / "decision.json"
            report_file = Path(temp_dir) / "pipeline_status.json"

            decision_file.write_text(json.dumps({
                "decision": "MONITOR_AND_COLLECT_DATA"
            }))

            with patch.object(
                run_pipeline,
                "DECISION_FILE",
                decision_file
            ), patch.object(
                run_pipeline,
                "PIPELINE_REPORT_FILE",
                report_file
            ), patch.object(
                run_pipeline,
                "validate_step_output",
                return_value="SUCCESS"
            ), patch.object(
                run_pipeline.subprocess,
                "run"
            ) as mock_run:

                result = run_pipeline.run_pipeline()

                self.assertEqual(result, 0)
                self.assertEqual(
                    mock_run.call_count,
                    len(run_pipeline.STEPS)
                )

                executed_scripts = [
                    call.args[0][1]
                    for call in mock_run.call_args_list
                ]

                expected_scripts = [
                    str(run_pipeline.ROOT / script)
                    for _, script in run_pipeline.STEPS
                ]

                self.assertEqual(executed_scripts, expected_scripts)

                for call in mock_run.call_args_list:
                    self.assertTrue(call.kwargs["check"])
                    self.assertEqual(call.kwargs["cwd"], run_pipeline.ROOT)

                self.assertTrue(report_file.exists())

    def test_stale_hourly_data_does_not_block_daily_prediction_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = Path(temp_dir) / "pipeline_status.json"

            def simulated_execute(name, script):
                if name == "Hourly Data Collection":
                    return "STALE_DATA"
                return "SUCCESS"

            with patch.object(
                run_pipeline,
                "PIPELINE_REPORT_FILE",
                report_file
            ), patch.object(
                run_pipeline,
                "execute_step",
                side_effect=simulated_execute
            ) as mock_execute:

                result = run_pipeline.run_pipeline()

                self.assertEqual(result, 2)

                executed_stages = [
                    call.args[0]
                    for call in mock_execute.call_args_list
                ]

                # Daily prediction chain must continue.
                daily_stages = [
                    "Daily Preprocessing",
                    "AQI Calculation",
                    "AQI Prediction",
                    "Daily Data Drift Detection",
                    "Prediction Performance Monitoring",
                ]

                for stage in daily_stages:
                    self.assertIn(stage, executed_stages)

                # Hourly-dependent stages must not execute.
                blocked_stages = [
                    "Hourly Preprocessing",
                    "Hourly Anomaly Detection",
                    "Pollution Episode Aggregation",
                    "Intelligent Retraining Decision",
                ]

                for stage in blocked_stages:
                    self.assertNotIn(stage, executed_stages)

                openaq_stages = [
                    "OpenAQ Historical Hourly Refresh",
                    "OpenAQ Hourly AQI Processing",
                    "OpenAQ Hourly Anomaly Detection",
                    "OpenAQ Pollution Episode Aggregation",
                ]

                for stage in openaq_stages:
                    self.assertIn(stage, executed_stages)

                self.assertEqual(
                    run_pipeline.STAGE_DEPENDENCIES["AQI Calculation"],
                    ["Daily Preprocessing"]
                )

                self.assertTrue(report_file.exists())

                report = json.loads(report_file.read_text())

                stage_status = {
                    item["stage"]: item["status"]
                    for item in report["stages"]
                }

                self.assertEqual(
                    stage_status["Hourly Data Collection"],
                    "STALE_DATA"
                )
                self.assertEqual(
                    stage_status["AQI Prediction"],
                    "SUCCESS"
                )
                self.assertEqual(
                    stage_status["Hourly Preprocessing"],
                    "SKIPPED_STALE_INPUT"
                )
                self.assertEqual(
                    stage_status["Intelligent Retraining Decision"],
                    "SKIPPED_STALE_INPUT"
                )

    def test_pipeline_stops_after_core_stage_failure(self):
        failure = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "failed_stage.py"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = Path(temp_dir) / "pipeline_status.json"

            with patch.object(
                run_pipeline,
                "PIPELINE_REPORT_FILE",
                report_file
            ), patch.object(
                run_pipeline,
                "validate_step_output",
                return_value="SUCCESS"
            ), patch.object(
                run_pipeline.subprocess,
                "run",
                side_effect=[None, None, None, None, failure]
            ) as mock_run:

                result = run_pipeline.run_pipeline()

                self.assertEqual(result, 1)
                self.assertEqual(mock_run.call_count, 5)
                self.assertTrue(report_file.exists())

    def test_old_daily_dataset_is_marked_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "old_daily.csv"

            pd.DataFrame({
                "period_start": ["2026-11-08"],
                "PM2.5": [40]
            }).to_csv(data_file, index=False)

            with patch.object(
                run_pipeline,
                "DAILY_DATA_FILE",
                data_file
            ):
                status = run_pipeline.validate_step_output(
                    "Daily Data Collection"
                )

            self.assertEqual(status, "STALE_DATA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
