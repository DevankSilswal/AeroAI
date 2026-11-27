from pathlib import Path
import json
import pandas as pd
from datetime import datetime

INPUT_FILE = Path("data/incoming/dehradun_latest.csv")
OUTPUT_FILE = Path("data/incoming/dehradun_daily.csv")
REPORT_FILE = Path("data/monitoring/daily_unit_validation.json")

EXPECTED_UNITS = {
    "AT": "°C",
    "CO": "mg/m³",
    "NO2": "µg/m³",
    "Ozone": "µg/m³",
    "PM10": "µg/m³",
    "PM2.5": "µg/m³",
    "RH": "%",
    "SO2": "µg/m³",
}

REQUIRED_POLLUTANTS = list(EXPECTED_UNITS)


def preprocess_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "station_id", "parameter_name", "period_start", "mean", "unit"
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["parameter_name"] = df["parameter_name"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()

    df = df[df["parameter_name"].isin(REQUIRED_POLLUTANTS)].copy()

    if df.empty:
        raise ValueError("No expected pollutant observations found.")

    unit_report = {}
    errors = []

    for pollutant, expected_unit in EXPECTED_UNITS.items():
        observed = sorted(
            df.loc[df["parameter_name"] == pollutant, "unit"].unique().tolist()
        )

        unit_report[pollutant] = {
            "expected_unit": expected_unit,
            "observed_units": observed,
            "status": "PASS" if observed == [expected_unit] else "FAIL",
        }

        if observed != [expected_unit]:
            errors.append(
                f"{pollutant}: expected {expected_unit}, observed {observed}"
            )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "source": str(INPUT_FILE),
        "status": "PASS" if not errors else "FAIL",
        "unit_validation": unit_report,
        "errors": errors,
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== UNIT VALIDATION ==========")
    for pollutant, result in unit_report.items():
        print(
            f"{pollutant}: {result['status']} "
            f"| Expected: {result['expected_unit']} "
            f"| Observed: {result['observed_units']}"
        )

    if errors:
        raise ValueError(
            "Unit validation failed. Daily preprocessing stopped:\n"
            + "\n".join(errors)
        )

    df["period_start"] = pd.to_datetime(
        df["period_start"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

    df = df.dropna(
        subset=["period_start", "mean", "parameter_name"]
    )

    daily = df.pivot_table(
        index="period_start",
        columns="parameter_name",
        values="mean",
        aggfunc="mean"
    ).reset_index()

    daily.columns.name = None

    for pollutant in REQUIRED_POLLUTANTS:
        if pollutant not in daily.columns:
            daily[pollutant] = pd.NA

    daily = daily[["period_start"] + REQUIRED_POLLUTANTS]
    daily = daily.sort_values("period_start")

    print("\n========== DATA COMPLETENESS ==========")
    print("Daily rows:", len(daily))
    print("\nMissing values:")
    print(daily[REQUIRED_POLLUTANTS].isna().sum().to_string())

    complete = daily.dropna(subset=REQUIRED_POLLUTANTS).copy()

    if complete.empty:
        raise ValueError("No complete daily observations available.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(OUTPUT_FILE, index=False)

    print("\n========== PREPROCESSING RESULT ==========")
    print("Unit validation: PASS")
    print("Complete daily rows:", len(complete))
    print("Date range:", complete["period_start"].min(), "to",
          complete["period_start"].max())
    print("Saved:", OUTPUT_FILE)
    print("Unit report:", REPORT_FILE)
    print("\nPreview:")
    print(complete.head().to_string(index=False))


if __name__ == "__main__":
    preprocess_data()
