import numpy as np
import pandas as pd


# CPCB National AQI concentration breakpoints.
# Units: CO = mg/m3; all other pollutants = ug/m3.
BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 600, 400, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400),
        (400, 1000, 400, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 380, 100, 200),
        (380, 800, 200, 300),
        (800, 1600, 300, 400),
        (1600, 2000, 400, 500),
    ],
    "Ozone": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 168, 100, 200),
        (168, 208, 200, 300),
        (208, 748, 300, 400),
        (748, 1000, 400, 500),
    ],
    "CO": [
        (0, 1, 0, 50),
        (1, 2, 50, 100),
        (2, 10, 100, 200),
        (10, 17, 200, 300),
        (17, 34, 300, 400),
        (34, 50, 400, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (200, 400, 50, 100),
        (400, 800, 100, 200),
        (800, 1200, 200, 300),
        (1200, 1800, 300, 400),
        (1800, 2400, 400, 500),
    ],
    "Pb": [
        (0, 0.5, 0, 50),
        (0.5, 1, 50, 100),
        (1, 2, 100, 200),
        (2, 3, 200, 300),
        (3, 3.5, 300, 400),
        (3.5, 5, 400, 500),
    ],
}

POLLUTANTS_24H = ["PM2.5", "PM10", "NO2", "SO2", "NH3", "Pb"]
POLLUTANTS_8H = ["CO", "Ozone"]
PM_REQUIRED = ["PM2.5", "PM10"]

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH",
    "PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI"
]

AQI_CATEGORIES = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]


def sub_index(value, pollutant):
    """Calculate one pollutant's CPCB breakpoint sub-index."""

    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")

    if pd.isna(value):
        return np.nan

    value = float(value)

    if not np.isfinite(value) or value < 0:
        return np.nan

    ranges = BREAKPOINTS[pollutant]

    for low, high, aqi_low, aqi_high in ranges:
        if low <= value <= high:
            result = (
                (aqi_high - aqi_low)
                * (value - low)
                / (high - low)
                + aqi_low
            )
            return int(np.rint(result))

    if value > ranges[-1][1]:
        return 500

    return np.nan


def aqi_category(value):
    if pd.isna(value):
        return "Insufficient data"

    value = float(value)

    for upper, category in AQI_CATEGORIES:
        if value <= upper:
            return category

    return "Severe"


def _calculate_final_aqi(df):
    """Apply CPCB minimum-three-pollutant and PM condition."""

    sub_columns = [p + "_AQI" for p in BREAKPOINTS]
    valid_subindices = df[sub_columns].notna()

    df["valid_pollutant_count"] = valid_subindices.sum(axis=1)

    pm_available = (
        df["PM2.5_AQI"].notna() | df["PM10_AQI"].notna()
    )

    enough_pollutants = df["valid_pollutant_count"] >= 3
    valid_overall = enough_pollutants & pm_available

    df["CPCB_AQI"] = df[sub_columns].max(axis=1)
    df.loc[~valid_overall, "CPCB_AQI"] = np.nan

    df["AQI_status"] = np.where(
        valid_overall,
        "VALID",
        "INSUFFICIENT_DATA"
    )

    df["AQI_category"] = df["CPCB_AQI"].apply(aqi_category)

    def dominant(row):
        values = row[sub_columns].dropna()
        if values.empty:
            return None
        return values.idxmax().replace("_AQI", "")

    df["dominant_pollutant"] = df.apply(dominant, axis=1)

    # Backward compatibility for existing AeroAI modules.
    # New code should use CPCB_AQI.
    df["estimated_AQI"] = df["CPCB_AQI"]

    return df


def _calculate_daily(df):
    """
    Daily-source mode.

    Calculates only pollutants whose CPCB averaging period is 24 hours.
    Daily CO and Ozone means are NOT valid substitutes for maximum 8-hour
    averages, so their sub-indices remain unavailable.

    Daily-source coverage cannot be verified unless hourly coverage
    information is supplied separately.
    """

    result = df.copy()

    for pollutant in BREAKPOINTS:
        result[pollutant + "_AQI"] = np.nan

    for pollutant in POLLUTANTS_24H:
        if pollutant in result.columns:
            values = pd.to_numeric(
                result[pollutant], errors="coerce"
            )
            result[pollutant + "_AQI"] = values.apply(
                lambda x: sub_index(x, pollutant)
            )

    result = _calculate_final_aqi(result)

    result["AQI_status"] = result["AQI_status"].replace(
        {"VALID": "VALID_METHOD_COVERAGE_UNVERIFIED"}
    )

    result["AQI_method"] = (
        "CPCB breakpoints; daily source; hourly coverage unverified"
    )

    return result


def _calculate_hourly(df, timestamp_col):
    """
    Calculate running AQI from hourly observations.

    24-hour pollutants: trailing 24-hour mean.
    CO and Ozone: maximum trailing 8-hour mean in the trailing 24 hours.
    At least 16 valid hourly observations in the trailing 24 hours
    are required for each pollutant sub-index.
    """

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Hourly mode requires timestamp column: {timestamp_col}"
        )

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(
        result[timestamp_col], errors="coerce"
    )

    if result[timestamp_col].isna().any():
        raise ValueError("Invalid timestamps found in hourly AQI input.")

    result = result.sort_values(timestamp_col).reset_index(drop=True)

    if result[timestamp_col].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. Aggregate duplicates before AQI calculation."
        )

    indexed = result.set_index(timestamp_col)

    for pollutant in BREAKPOINTS:
        result[pollutant + "_AQI"] = np.nan

    for pollutant in POLLUTANTS_24H:
        if pollutant not in indexed.columns:
            continue

        values = pd.to_numeric(
            indexed[pollutant], errors="coerce"
        )

        rolling_mean = values.rolling(
            "24h", min_periods=1, closed="right"
        ).mean()

        valid_hours = values.rolling(
            "24h", min_periods=1, closed="right"
        ).count()

        valid_mean = rolling_mean.where(valid_hours >= 16)

        result[pollutant + "_24h_avg"] = valid_mean.to_numpy()
        result[pollutant + "_AQI"] = valid_mean.apply(
            lambda x: sub_index(x, pollutant)
        ).to_numpy()

    for pollutant in POLLUTANTS_8H:
        if pollutant not in indexed.columns:
            continue

        values = pd.to_numeric(
            indexed[pollutant], errors="coerce"
        )

        eight_hour_mean = values.rolling(
            "8h", min_periods=8, closed="right"
        ).mean()

        max_eight_hour_mean = eight_hour_mean.rolling(
            "24h", min_periods=1, closed="right"
        ).max()

        valid_hours = values.rolling(
            "24h", min_periods=1, closed="right"
        ).count()

        valid_mean = max_eight_hour_mean.where(valid_hours >= 16)

        result[pollutant + "_max_8h_avg"] = valid_mean.to_numpy()
        result[pollutant + "_AQI"] = valid_mean.apply(
            lambda x: sub_index(x, pollutant)
        ).to_numpy()

    result = _calculate_final_aqi(result)
    result["AQI_method"] = (
        "CPCB breakpoints; rolling 24h and maximum rolling 8h averages"
    )

    return result


def calculate_aqi(data, mode="auto", timestamp_col="period_start"):
    """
    Shared AeroAI AQI calculator.

    mode='hourly': CPCB running-average calculation from hourly readings.
    mode='daily': 24-hour pollutants only; coverage remains unverified.
    mode='auto': infer daily/hourly from timestamp spacing.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("calculate_aqi expects a pandas DataFrame.")

    if data.empty:
        raise ValueError("Cannot calculate AQI from an empty DataFrame.")

    if mode not in {"auto", "hourly", "daily"}:
        raise ValueError("mode must be auto, hourly, or daily.")

    selected_mode = mode

    if selected_mode == "auto":
        if timestamp_col not in data.columns:
            selected_mode = "daily"
        else:
            timestamps = pd.to_datetime(
                data[timestamp_col], errors="coerce"
            ).dropna().sort_values()

            if len(timestamps) >= 2:
                median_gap = timestamps.diff().dropna().median()
                selected_mode = (
                    "hourly"
                    if median_gap <= pd.Timedelta(hours=2)
                    else "daily"
                )
            else:
                selected_mode = "daily"

    if selected_mode == "hourly":
        return _calculate_hourly(data, timestamp_col)

    return _calculate_daily(data)


# ---------------------------------------------------------
# DAILY PIPELINE EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import json
    from datetime import datetime

    ROOT = Path(__file__).resolve().parent.parent

    INPUT_FILE = ROOT / "data/incoming/dehradun_daily.csv"
    OUTPUT_FILE = ROOT / "data/incoming/dehradun_aqi_processed.csv"
    REPORT_FILE = ROOT / "data/monitoring/daily_cpcb_aqi_report.json"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Daily preprocessed data not found: {INPUT_FILE}"
        )

    daily_data = pd.read_csv(INPUT_FILE)

    result = calculate_aqi(
        daily_data,
        mode="daily"
    )

    if "CPCB_AQI" not in result.columns:
        raise RuntimeError("CPCB_AQI calculation failed.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": str(INPUT_FILE),
        "output": str(OUTPUT_FILE),
        "method": "CPCB breakpoint calculation using daily source",
        "hourly_coverage_verified": False,
        "rows_processed": int(len(result)),
        "valid_cpcb_aqi_rows": int(result["CPCB_AQI"].notna().sum()),
        "insufficient_data_rows": int(result["CPCB_AQI"].isna().sum()),
        "status_counts": {
            str(k): int(v)
            for k, v in result["AQI_status"].value_counts().items()
        },
        "target_column": "CPCB_AQI",
        "legacy_alias": "estimated_AQI",
        "limitations": [
            "Daily source does not verify minimum hourly coverage.",
            "Daily CO and Ozone means are excluded from CPCB sub-index calculation.",
            "NH3 and Pb are unavailable in this source."
        ]
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\n========== CPCB DAILY AQI CALCULATION ==========")
    print("Rows processed:", len(result))
    print("Valid CPCB AQI:", report["valid_cpcb_aqi_rows"])
    print("Insufficient data:", report["insufficient_data_rows"])
    print("\nStatus counts:")
    print(result["AQI_status"].value_counts().to_string())
    print("\nAQI preview:")
    print(
        result[
            ["period_start", "PM2.5_AQI", "PM10_AQI",
             "NO2_AQI", "SO2_AQI", "CPCB_AQI",
             "AQI_category", "AQI_status"]
        ].tail().to_string(index=False)
    )
    print("\nSaved:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)
