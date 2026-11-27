import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.weather_data import get_weather, weather_description

# ============================================================
# AEROAI - Air Quality Intelligence Platform
# Academic prototype: estimated AQI, not official CPCB AQI.
# ============================================================

ROOT = Path(__file__).resolve().parent

HISTORICAL_FILE = ROOT / "data/raw/dehradun_aqi_dataset.csv"
TRAINING_SOURCE_FILE = ROOT / "data/raw/dehradun_cleaned.csv"
MODEL_FILE = ROOT / "best_aqi_model.pkl"
RESULTS_FILE = ROOT / "model_results.json"

LATEST_FILE = ROOT / "data/incoming/dehradun_aqi_processed.csv"
DAILY_FILE = ROOT / "data/incoming/dehradun_daily.csv"
PREDICTION_FILE = ROOT / "data/predictions/prediction_history.csv"
EVALUATION_FILE = ROOT / "data/monitoring/prediction_evaluation.csv"
MONITORING_FILE = ROOT / "data/monitoring/model_monitoring.json"
DRIFT_FILE = ROOT / "data/monitoring/drift_report.csv"
DRIFT_SUMMARY_FILE = ROOT / "data/monitoring/drift_summary.json"
RETRAINING_FILE = ROOT / "data/monitoring/retraining_report.json"
LIVE_AQI_FILE = ROOT / "data/incoming/current_aqi_snapshot.json"
OPENAQ_FILE = ROOT / "data/incoming/openaq_snapshot.json"

# Adaptive Intelligence artifacts
HOURLY_RAW_FILE = ROOT / "data/incoming/dehradun_hourly_raw.csv"
HOURLY_PROCESSED_FILE = ROOT / "data/incoming/dehradun_hourly_processed.csv"
HOURLY_QUALITY_FILE = ROOT / "data/monitoring/hourly_data_quality.json"
HOURLY_ANOMALY_FILE = ROOT / "data/monitoring/hourly_anomaly_report.csv"
HOURLY_EVENTS_FILE = ROOT / "data/monitoring/hourly_anomaly_events.json"
RETRAINING_DECISION_FILE = ROOT / "data/monitoring/retraining_decision.json"
PIPELINE_SCRIPT = ROOT / "src/run_pipeline.py"
MLFLOW_DB = ROOT / "mlflow.db"

FEATURES = [
    "PM2.5", "PM10", "NO2", "SO2", "AT", "RH",
    "PM2.5_AQI", "PM10_AQI", "NO2_AQI", "SO2_AQI"
]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AT", "RH"]

st.set_page_config(
    page_title="AeroAI | Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {background: #0b1220; color: #e5e7eb;}
[data-testid="stSidebar"] {background: #111c2e;}
.metric-card {
    background: linear-gradient(135deg,#14243a,#172f3d);
    border: 1px solid #29445b;
    border-radius: 13px;
    padding: 16px 18px;
    min-height: 105px;
}
.metric-label {color:#9fb3c8;font-size:12px;letter-spacing:.4px;}
.metric-value {color:#f1f5f9;font-size:26px;font-weight:700; margin-top:5px;}
.metric-note {color:#55d6a0;font-size:12px;margin-top:3px;}
h1,h2,h3 {color:#e8f1f8;}
div[data-testid="stMetric"] {
    background: #14243a; border: 1px solid #29445b;
    border-radius: 12px; padding: 12px;
}
.small-note {color:#9fb3c8;font-size:12px;}
</style>
""", unsafe_allow_html=True)


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.relative_to(ROOT)}: {exc}")
        return pd.DataFrame()


def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        st.warning(f"Could not read {path.relative_to(ROOT)}: {exc}")
        return {}


def load_model():
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return None


def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True
    )


def fmt_date(value, fmt="%d %b %Y"):
    try:
        return pd.to_datetime(value).strftime(fmt)
    except Exception:
        return "Not available"


def aqi_category(value):
    if pd.isna(value):
        return "Not available"
    value = float(value)
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


def latest_observations():
    current = read_csv(LATEST_FILE)
    if not current.empty and "period_start" in current.columns:
        current["period_start"] = pd.to_datetime(current["period_start"], errors="coerce")
        current = current.dropna(subset=["period_start"]).sort_values("period_start")
        return current, "API-retrieved historical observations"
    old = read_csv(HISTORICAL_FILE)
    if not old.empty and "period_start" in old.columns:
        old["period_start"] = pd.to_datetime(old["period_start"], errors="coerce")
        old = old.dropna(subset=["period_start"]).sort_values("period_start")
        return old, "Historical training dataset"
    return pd.DataFrame(), "No dataset available"


def run_pipeline():
    if not PIPELINE_SCRIPT.exists():
        return False, "src/run_pipeline.py not found."
    try:
        result = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output[-18000:]
    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 15 minutes."
    except Exception as exc:
        return False, str(exc)


# Load artifacts on each Streamlit rerun so newly generated reports appear.
historical = read_csv(HISTORICAL_FILE)
if not historical.empty and "period_start" in historical.columns:
    historical["period_start"] = pd.to_datetime(historical["period_start"], errors="coerce")

latest_df, latest_source_label = latest_observations()
if not latest_df.empty and "period_start" in latest_df.columns:
    latest_df = latest_df.dropna(subset=["period_start"])
results = read_json(RESULTS_FILE)
prediction_history = read_csv(PREDICTION_FILE)
evaluation = read_csv(EVALUATION_FILE)
monitoring = read_json(MONITORING_FILE)
drift = read_csv(DRIFT_FILE)
drift_summary = read_json(DRIFT_SUMMARY_FILE)
retraining = read_json(RETRAINING_FILE)
live_aqi_snapshot = read_json(LIVE_AQI_FILE)
openaq_snapshot = read_json(OPENAQ_FILE)

@st.cache_data(ttl=900, show_spinner=False)
def load_weather_snapshot():
    return get_weather()

weather_snapshot = load_weather_snapshot()

hourly_df = read_csv(HOURLY_PROCESSED_FILE)
hourly_quality = read_json(HOURLY_QUALITY_FILE)
hourly_anomalies = read_csv(HOURLY_ANOMALY_FILE)
hourly_events = read_json(HOURLY_EVENTS_FILE)
adaptive_decision = read_json(RETRAINING_DECISION_FILE)

st.sidebar.title("🌿 AeroAI")
st.sidebar.caption("Air Quality Intelligence Platform")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Air Quality Analysis",
        "AQI Prediction",
        "Model Performance",
        "Data Monitoring",
        "Adaptive Intelligence Center",
        "Intelligent Retraining",
        "MLflow Experiment Tracking",
        "Pipeline Status"
    ]
)
st.sidebar.divider()
st.sidebar.caption("Location: Dehradun, Uttarakhand")
if not latest_df.empty:
    latest_date_sidebar = latest_df["period_start"].max()
    st.sidebar.caption(f"Latest available observation: {fmt_date(latest_date_sidebar)}")
st.sidebar.caption("Historical/API-retrieved data — not live monitoring")
st.sidebar.caption("Estimated AQI — not official CPCB AQI")

# Common latest observation and prediction information.
latest_row = latest_df.iloc[-1] if not latest_df.empty else None
latest_aqi = (
    float(latest_row["estimated_AQI"])
    if latest_row is not None and "estimated_AQI" in latest_row and pd.notna(latest_row["estimated_AQI"])
    else None
)
latest_date = latest_row["period_start"] if latest_row is not None else None

model = load_model()
model_name = results.get("best_model", "Unknown")
next_prediction = None
prediction_error = None

if model is not None and latest_row is not None:
    missing_features = [col for col in FEATURES if col not in latest_df.columns]
    if not missing_features:
        try:
            input_row = pd.DataFrame([[latest_row[col] for col in FEATURES]], columns=FEATURES)
            if input_row.notna().all(axis=None):
                next_prediction = float(model.predict(input_row)[0])
                next_prediction = float(np.clip(next_prediction, 0, 500))
            else:
                prediction_error = "Latest observation has missing model features."
        except Exception as exc:
            prediction_error = str(exc)
    else:
        prediction_error = f"Latest dataset is missing features: {missing_features}"

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("🌿 AeroAI Dashboard")
    st.caption("AI-enabled AQI estimation, next-day forecasting and model monitoring")

    # CURRENT CPCB STATION OBSERVATION
    st.subheader("Current Air Quality — CPCB Monitoring Station")

    if live_aqi_snapshot:

        station_name = live_aqi_snapshot.get("station_name", "Unknown station")
        station_aqi = live_aqi_snapshot.get("aqi")
        station_category = live_aqi_snapshot.get("category", "Unavailable")
        station_updated = live_aqi_snapshot.get("station_updated_at")

        # Recalculate age on every dashboard rerun.
        age_hours = None

        try:
            updated_time = pd.to_datetime(station_updated, errors="coerce")

            if pd.notna(updated_time):
                if updated_time.tzinfo is None:
                    updated_time = updated_time.tz_localize("Asia/Kolkata")
                else:
                    updated_time = updated_time.tz_convert("Asia/Kolkata")

                current_time = pd.Timestamp.now(tz="Asia/Kolkata")
                age_hours = (current_time - updated_time).total_seconds() / 3600

        except Exception:
            age_hours = None

        station_fresh = (
            age_hours is not None
            and 0 <= age_hours <= 3
        )

        if station_fresh:
            st.success("Station feed is fresh.")
        elif age_hours is not None:
            st.warning(
                f"Station feed is stale or delayed. "
                f"Latest reported observation is {age_hours:.1f} hours old."
            )
        else:
            st.warning("Station timestamp could not be verified.")

        st.write(f"**Station:** {station_name}")

        live_cols = st.columns(4)

        live_cols[0].metric(
            "CPCB station AQI",
            station_aqi if station_aqi is not None else "N/A"
        )

        live_cols[1].metric(
            "Reported category",
            station_category
        )

        live_cols[2].metric(
            "Main pollutant",
            live_aqi_snapshot.get("main_pollutant", "N/A")
        )

        live_cols[3].metric(
            "Feed age",
            f"{age_hours:.1f} hours" if age_hours is not None else "Unknown"
        )

        st.caption(
            f"Last station update: {station_updated or 'Unavailable'}"
        )

        # Pollutant values from this API are sub-indices, not raw concentrations.
        st.markdown("**Pollutant sub-indices**")

        subindices = live_aqi_snapshot.get("pollutant_subindices", {})

        if subindices:
            rows = []

            for pollutant, values in subindices.items():
                rows.append({
                    "Pollutant": pollutant,
                    "Average sub-index": values.get("average_subindex"),
                    "Minimum sub-index": values.get("minimum"),
                    "Maximum sub-index": values.get("maximum")
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These values are pollutant sub-indices, not measured "
                "concentrations in µg/m³ or ppb."
            )

        # Nearby station map and station comparison.
        st.markdown("**Nearby returned CPCB monitoring stations**")

        station_rows = live_aqi_snapshot.get("nearby_stations", [])

        if station_rows:

            stations = pd.DataFrame(station_rows)

            for column in ["latitude", "longitude", "distance_km", "aqi"]:
                if column in stations.columns:
                    stations[column] = pd.to_numeric(
                        stations[column], errors="coerce"
                    )

            map_data = stations.dropna(
                subset=["latitude", "longitude"]
            )

            if not map_data.empty:
                map_data = map_data.rename(
                    columns={
                        "latitude": "lat",
                        "longitude": "lon"
                    }
                )

                st.map(
                    map_data[["lat", "lon"]],
                    zoom=6
                )

            display_columns = [
                "station", "city", "state", "aqi",
                "category", "distance_km", "updated"
            ]

            display_columns = [
                column for column in display_columns
                if column in stations.columns
            ]

            st.dataframe(
                stations[display_columns],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "The API may return stations outside Dehradun. "
                "Distance and city are shown to distinguish regional stations."
            )

    else:
        st.warning(
            "Current station snapshot is unavailable. "
            "Run `python src/live_air_quality.py` to retrieve station data. "
            "Historical AQI must not be interpreted as current AQI."
        )

    attribution = live_aqi_snapshot.get("attribution")

    if attribution:
        st.caption(f"Data attribution: {attribution}")

    source_url = live_aqi_snapshot.get("source_url")

    if source_url:
        st.caption(f"Source dataset: {source_url}")

    # OPENAQ LIVE SENSOR DATA
    st.divider()
    st.subheader("🌍 Live Air Quality Sensors — OpenAQ")
    st.caption(
        "Raw measurements from the Doon University CPCB-linked station. "
        "These are sensor concentrations, not an official AQI value."
    )

    if openaq_snapshot and openaq_snapshot.get("readings"):
        st.write(
            f"**Station:** {openaq_snapshot.get('station_name', 'Unknown')}"
        )

        readings = openaq_snapshot.get("readings", {})
        current_time = pd.Timestamp.now(tz="Asia/Kolkata")

        sensor_labels = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO₂",
            "so2": "SO₂",
            "co": "CO",
            "o3": "O₃",
            "no": "NO",
            "nox": "NOx",
            "temperature": "Temperature",
            "relativehumidity": "Humidity",
            "wind_speed": "Wind speed",
            "wind_direction": "Wind direction"
        }

        def openaq_reading_status(reading):
            timestamp = pd.to_datetime(
                reading.get("timestamp"), errors="coerce"
            )
            if pd.isna(timestamp):
                return None, "Unknown"

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("Asia/Kolkata")
            else:
                timestamp = timestamp.tz_convert("Asia/Kolkata")

            age = (current_time - timestamp).total_seconds() / 3600

            if age < 0:
                return age, "Invalid future timestamp"
            if age <= 3:
                return age, "Fresh"
            return age, "Stale"

        def openaq_display_value(key):
            reading = readings.get(key)
            if not isinstance(reading, dict):
                return "N/A"

            value = reading.get("value")
            unit = reading.get("unit", "")

            if value is None:
                return "N/A"

            try:
                return f"{float(value):g} {unit}".strip()
            except (ValueError, TypeError):
                return "N/A"

        # Prominent pollutant measurements
        main_sensors = [
            ("pm25", "PM2.5"),
            ("pm10", "PM10"),
            ("no2", "NO₂"),
            ("so2", "SO₂")
        ]

        metric_cols = st.columns(4)

        for col, (key, label) in zip(metric_cols, main_sensors):
            reading = readings.get(key, {})
            age, status = openaq_reading_status(reading)

            col.metric(label, openaq_display_value(key))
            if status == "Fresh":
                col.caption(f"🟢 Fresh · {age:.1f} hours old")
            elif status == "Stale":
                col.caption(f"🟠 Stale · {age:.1f} hours old")
            else:
                col.caption(f"⚪ {status}")

        # Additional sensor readings
        with st.expander("View all OpenAQ sensor measurements", expanded=True):
            rows = []

            for key, reading in readings.items():
                if not isinstance(reading, dict):
                    continue

                age, status = openaq_reading_status(reading)

                rows.append({
                    "Sensor": sensor_labels.get(key, key),
                    "Value": reading.get("value"),
                    "Unit": reading.get("unit", ""),
                    "Status": status,
                    "Age (hours)": round(age, 2) if age is not None else None,
                    "Last updated": reading.get("timestamp", "N/A"),
                    "Sensor ID": reading.get("sensor_id", "N/A")
                })

            if rows:
                sensor_df = pd.DataFrame(rows)
                st.dataframe(
                    sensor_df,
                    use_container_width=True,
                    hide_index=True
                )

        fresh_count = sum(
            1 for reading in readings.values()
            if isinstance(reading, dict)
            and openaq_reading_status(reading)[1] == "Fresh"
        )
        stale_count = sum(
            1 for reading in readings.values()
            if isinstance(reading, dict)
            and openaq_reading_status(reading)[1] == "Stale"
        )

        st.caption(
            f"Sensor freshness: {fresh_count} fresh · "
            f"{stale_count} stale · {len(readings)} total readings"
        )
        st.caption(
            f"Snapshot retrieved: {openaq_snapshot.get('retrieved_at', 'Unknown')}"
        )
        st.caption(
            openaq_snapshot.get(
                "note",
                "Raw sensor readings; not an official CPCB AQI."
            )
        )

        st.markdown(
            "[View Doon University station on OpenAQ]"
            "(https://explore.openaq.org/locations/317035)"
        )
        st.caption("Source: OpenAQ API v3 · Location ID 317035")

    else:
        st.warning(
            "OpenAQ snapshot unavailable. Run "
            "`python src/openaq_data.py` to retrieve sensor readings."
        )

    # CURRENT WEATHER — OPEN-METEO
    st.divider()
    st.subheader("🌤️ Current Weather — Dehradun")
    st.caption(
        "Weather-model data for the Doon University area. "
        "This is separate from CPCB air-quality measurements."
    )

    if weather_snapshot.get("status") in ("success", "cached"):
        weather = weather_snapshot.get("current", {})
        daily_weather = weather_snapshot.get("daily", {})

        if weather_snapshot.get("status") == "cached":
            st.warning("Weather API unavailable; displaying the last cached weather snapshot.")
        else:
            st.success("Weather data retrieved successfully.")

        condition = weather_description(weather.get("weather_code"))
        st.markdown(f"**{condition}**")

        weather_cols = st.columns(4)
        weather_cols[0].metric(
            "Temperature",
            f"{weather['temperature_2m']:.1f} °C",
            f"Feels like {weather['apparent_temperature']:.1f} °C"
        )
        weather_cols[1].metric(
            "Humidity",
            f"{weather['relative_humidity_2m']}%"
        )
        weather_cols[2].metric(
            "Wind speed",
            f"{weather['wind_speed_10m']:.1f} km/h",
            f"Direction {weather.get('wind_direction_10m', 'N/A')}°"
        )
        weather_cols[3].metric(
            "Precipitation",
            f"{weather.get('precipitation', 0)} mm"
        )

        extra_cols = st.columns(4)
        extra_cols[0].metric(
            "Cloud cover",
            f"{weather.get('cloud_cover', 'N/A')}%"
        )
        extra_cols[1].metric(
            "Today's max UV",
            daily_weather.get("uv_index_max", "N/A")
        )
        extra_cols[2].metric(
            "Today's maximum",
            f"{daily_weather['temperature_max']:.1f} °C"
            if daily_weather.get("temperature_max") is not None else "N/A"
        )
        extra_cols[3].metric(
            "Today's minimum",
            f"{daily_weather['temperature_min']:.1f} °C"
            if daily_weather.get("temperature_min") is not None else "N/A"
        )

        st.caption(
            f"Weather model update: {weather.get('time', 'Not available')} "
            f"({weather_snapshot.get('timezone', 'local time')})"
        )
        st.caption(
            f"Source: Open-Meteo · {weather_snapshot.get('source_url', 'https://open-meteo.com/')}"
        )
        if weather_snapshot.get("error"):
            st.caption(f"Latest refresh issue: {weather_snapshot['error']}")
    else:
        st.warning(
            "Weather data is currently unavailable. "
            f"Details: {weather_snapshot.get('error', 'Unknown error')}"
        )

    st.divider()


    if latest_row is None:
        st.error("No usable observation dataset found. Run the pipeline first.")
        st.stop()

    a, b, c, d = st.columns(4)
    with a:
        metric_card("LATEST HISTORICAL ESTIMATED AQI", f"{latest_aqi:.1f}" if latest_aqi is not None else "N/A",
                    aqi_category(latest_aqi))
    with b:
        metric_card("NEXT-DAY MODEL FORECAST", f"{next_prediction:.1f}" if next_prediction is not None else "N/A",
                    aqi_category(next_prediction) if next_prediction is not None else "Prediction unavailable")
    with c:
        metric_card("ACTIVE MODEL", model_name, "Loaded from saved model artifact")
    with d:
        metric_card("LATEST DATA ROWS", f"{len(latest_df):,}", latest_source_label)

    st.caption(
        "Historical AQI estimate and model forecast use the latest available "
        "daily dataset. They are separate from the CPCB station observation above."
    )

    st.subheader("Latest available AQI trend")
    if "estimated_AQI" in latest_df.columns:
        fig = px.line(
            latest_df, x="period_start", y="estimated_AQI",
            labels={"period_start": "Date", "estimated_AQI": "Estimated AQI"},
            title="Daily partial-pollutant AQI estimate"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AQI trend is not available in the selected dataset.")

    st.subheader("Latest available pollutant readings")
    available_pollutants = [c for c in POLLUTANTS if c in latest_df.columns]
    if available_pollutants:
        cols = st.columns(min(4, len(available_pollutants)))
        for index, pollutant in enumerate(available_pollutants):
            value = latest_row[pollutant]
            cols[index % len(cols)].metric(pollutant, f"{value:.2f}" if pd.notna(value) else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Data drift")
        drift_status = drift_summary.get("overall_status", "Report unavailable")
        st.write(drift_status)
        if "High" in str(drift_status):
            st.warning("Distribution drift detected. Investigate the feature shifts; drift alone does not prove model failure.")
        elif drift_summary:
            st.info("Drift report loaded.")
        else:
            st.info("Run the pipeline to generate a drift report.")
    with right:
        st.subheader("Prediction monitoring")
        st.write(monitoring.get("status", "Monitoring report unavailable"))
        st.metric("Pending predictions", monitoring.get("pending_predictions", "N/A"))

    st.info(
        "AQI is a locally calculated partial-pollutant estimate. It is not an official CPCB AQI value. "
        "The latest API data is historical and may lag behind the current date."
    )

# ---------------- AIR QUALITY ANALYSIS ----------------
elif page == "Air Quality Analysis":
    st.title("Air Quality Analysis")
    st.caption(f"Source: {latest_source_label}")

    if latest_df.empty:
        st.warning("No observation data found.")
        st.stop()

    min_date = latest_df["period_start"].min().date()
    max_date = latest_df["period_start"].max().date()
    date_range = st.date_input(
        "Select date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        view = latest_df[
            (latest_df["period_start"].dt.date >= date_range[0]) &
            (latest_df["period_start"].dt.date <= date_range[1])
        ]
    else:
        view = latest_df.copy()

    if "estimated_AQI" in view.columns:
        st.subheader("Estimated AQI over time")
        fig = px.line(view, x="period_start", y="estimated_AQI",
                      labels={"period_start": "Date", "estimated_AQI": "Estimated AQI"})
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    pollutants = [c for c in POLLUTANTS if c in view.columns]
    selected = st.multiselect("Choose pollutants", pollutants, default=pollutants[:2])
    if selected:
        fig2 = px.line(
            view, x="period_start", y=selected,
            labels={"period_start": "Date", "value": "Daily mean", "variable": "Pollutant"}
        )
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(view.tail(30), use_container_width=True)
    st.download_button(
        "Download displayed data as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="aeroai_air_quality_data.csv",
        mime="text/csv"
    )

# ---------------- AQI PREDICTION ----------------
elif page == "AQI Prediction":
    st.title("Next-Day AQI Prediction")
    st.caption("Forecast based on the latest complete observation and saved model.")

    if latest_date is not None:
        st.warning(
            f"Input observation date: {fmt_date(latest_date, '%d %B %Y')}. "
            "This is a forecast, not a live AQI reading."
        )

    if prediction_error:
        st.error(prediction_error)
    elif next_prediction is not None:
        a, b = st.columns(2)
        a.metric("Predicted next-day estimated AQI", f"{next_prediction:.2f}")
        b.metric("Model", model_name)
        st.write("Predicted category:", aqi_category(next_prediction))

        if st.button("Save current prediction", type="primary"):
            PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
            forecast_date = (pd.to_datetime(latest_date) + pd.Timedelta(days=1)).date()
            new_record = pd.DataFrame([{
                "source_date": str(pd.to_datetime(latest_date).date()),
                "forecast_date": str(forecast_date),
                "predicted_estimated_AQI": next_prediction,
                "model": model_name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "status": "Pending"
            }])
            if PREDICTION_FILE.exists():
                old = pd.read_csv(PREDICTION_FILE)
                if "forecast_date" in old.columns:
                    old = old[old["forecast_date"].astype(str) != str(forecast_date)]
                combined = pd.concat([old, new_record], ignore_index=True)
            else:
                combined = new_record
            combined.to_csv(PREDICTION_FILE, index=False)
            st.success(f"Saved forecast for {forecast_date}.")
            st.rerun()
    else:
        st.info("Prediction cannot be generated from the currently available data.")

    st.subheader("Prediction history")
    if not prediction_history.empty:
        st.dataframe(prediction_history.tail(30), use_container_width=True)
    else:
        st.info("No saved predictions yet. The automated pipeline creates prediction history.")

    st.subheader("Actual outcome evaluation")
    if not evaluation.empty:
        st.dataframe(evaluation.tail(30), use_container_width=True)
    else:
        st.info("No evaluation file yet. Prediction errors appear only after actual target-date observations arrive.")

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Chronological historical test-set results from the training run.")

    model_results = results.get("results", {})
    if model_results:
        comparison = pd.DataFrame([
            {"Model": name, **metrics} for name, metrics in model_results.items()
        ])
        st.dataframe(comparison, use_container_width=True)
        metric_options = [m for m in ["MAE", "RMSE", "R2"] if m in comparison.columns]
        if metric_options:
            metric = st.selectbox("Compare metric", metric_options)
            fig = px.bar(comparison, x="Model", y=metric, color="Model",
                         title=f"Model comparison — {metric}")
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model results file is missing or has no comparison metrics.")

    st.info(
        "Historical test metrics describe the 2024 chronological holdout only. "
        "They are not a verified estimate of prediction error on the newer API data."
    )

    if not evaluation.empty and "absolute_error" in evaluation.columns:
        observed = pd.to_numeric(evaluation["absolute_error"], errors="coerce").dropna()
        if len(observed):
            st.subheader("Observed forecast errors on newer observations")
            st.metric("Observed MAE", f"{observed.mean():.2f}")
            st.caption(f"Based on {len(observed)} evaluated forecast(s); small samples are not conclusive.")
        else:
            st.info("No actual forecast outcomes are available yet.")

# ---------------- DATA MONITORING ----------------
elif page == "Data Monitoring":
    st.title("Data & Drift Monitoring")
    st.caption("Reports generated by the automated pipeline.")

    if not latest_df.empty:
        missing_cells = int(latest_df.isna().sum().sum())
        duplicate_rows = int(latest_df.duplicated().sum())
        a, b, c = st.columns(3)
        a.metric("Observation rows", len(latest_df))
        b.metric("Missing cells", missing_cells)
        c.metric("Duplicate rows", duplicate_rows)

    st.subheader("Feature drift report")
    if not drift.empty:
        st.dataframe(drift, use_container_width=True)
        status_col = "status" if "status" in drift.columns else ("Status" if "Status" in drift.columns else None)
        if status_col:
            high = int(drift[status_col].astype(str).str.contains("High", case=False).sum())
            moderate = int(drift[status_col].astype(str).str.contains("Moderate", case=False).sum())
            x, y = st.columns(2)
            x.metric("High-drift features", high)
            y.metric("Moderate-drift features", moderate)
        st.caption(
            f"Reference rows: {drift_summary.get('reference_rows', 'N/A')} | "
            f"Current rows: {drift_summary.get('current_rows', 'N/A')}"
        )
    else:
        st.info("Drift report not found. Run the pipeline to generate it.")

    st.warning(
        "The current drift comparison uses only 13 recent daily observations. "
        "Treat the PSI result as an investigation signal, not proof of model degradation."
    )

    st.subheader("Prediction monitoring")
    if monitoring:
        x, y, z = st.columns(3)
        x.metric("Evaluated predictions", monitoring.get("evaluated_predictions", 0))
        y.metric("Pending predictions", monitoring.get("pending_predictions", 0))
        z.metric("Observed MAE", monitoring.get("observed_MAE") if monitoring.get("observed_MAE") is not None else "N/A")
        st.write(monitoring.get("status", "Status unavailable"))
    if not evaluation.empty:
        st.dataframe(evaluation.tail(30), use_container_width=True)

# ---------------- INTELLIGENT RETRAINING ----------------
elif page == "Intelligent Retraining":
    st.title("Intelligent Retraining")
    st.caption("Candidate model evaluation and guarded promotion decision.")

    if retraining:
        status = retraining.get("status", "Unknown")
        st.subheader("Latest retraining decision")
        st.write(status)
        a, b, c = st.columns(3)
        a.metric("Available labelled rows", retraining.get("available_rows", "N/A"))
        b.metric("Minimum required rows", retraining.get("minimum_required_rows", 90))
        c.metric("Active model changed", "Yes" if retraining.get("active_model_changed") else "No")
        if "champion_test_MAE" in retraining:
            x, y = st.columns(2)
            x.metric("Champion test MAE", retraining.get("champion_test_MAE"))
            y.metric("Candidate test MAE", retraining.get("candidate_test_MAE"))
        st.json(retraining)
    else:
        st.info("No retraining report found. Run the pipeline to generate the decision.")

    st.warning(
        "Current retraining is correctly blocked until at least 90 labelled next-day observations "
        "are available. A drift alert alone does not trigger model replacement."
    )


# ---------------- ADAPTIVE INTELLIGENCE CENTER ----------------

elif page == "Adaptive Intelligence Center":

    st.title("Adaptive Intelligence Center")
    st.caption(
        "Hourly pollution intelligence, anomaly episodes and "
        "condition-based model retraining decisions."
    )

    decision_name = adaptive_decision.get(
        "decision", "Decision report unavailable"
    )

    if decision_name == "WAIT_FOR_LABELLED_DATA":
        st.warning(
            "Retraining is waiting for sufficient labelled observations."
        )
    elif decision_name == "CANDIDATE_TRAINING_ELIGIBLE":
        st.info(
            "Candidate training is eligible. Model validation is still required."
        )
    elif decision_name == "MONITOR_AND_COLLECT_DATA":
        st.success(
            "No confirmed retraining trigger. Continue monitoring."
        )
    else:
        st.info("Run the adaptive pipeline to generate the latest decision.")

    # Main adaptive metrics
    total_hours = int(
        hourly_quality.get("hourly_timestamps", len(hourly_df))
    )
    complete_aqi = int(
        hourly_quality.get("complete_aqi_hours", 0)
    )
    complete_model = int(
        hourly_quality.get("complete_model_feature_hours", 0)
    )
    episodes_count = int(hourly_events.get("pollution_episodes", 0))

    anomaly_count = int(
        (hourly_anomalies.get("status", pd.Series(dtype=str))
         == "Potential pollution spike").sum()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Hourly timestamps", total_hours)
    c2.metric("Complete AQI hours", complete_aqi)
    c3.metric("Complete ML hours", complete_model)
    c4.metric("Feature alerts", anomaly_count)
    c5.metric("Pollution episodes", episodes_count)

    st.divider()

    # Hourly AQI proxy trend
    st.subheader("Hourly Estimated AQI Trend")

    if not hourly_df.empty and "estimated_AQI" in hourly_df.columns:

        chart_df = hourly_df.copy()

        chart_df["period_start"] = pd.to_datetime(
            chart_df["period_start"], errors="coerce"
        )

        chart_df["estimated_AQI"] = pd.to_numeric(
            chart_df["estimated_AQI"], errors="coerce"
        )

        if "aqi_data_complete" in chart_df.columns:
            valid = (
                chart_df["aqi_data_complete"]
                .astype(str).str.lower().isin(["true", "1"])
            )
            chart_df = chart_df[valid]

        chart_df = chart_df.dropna(
            subset=["period_start", "estimated_AQI"]
        ).sort_values("period_start")

        if not chart_df.empty:
            fig = px.line(
                chart_df,
                x="period_start",
                y="estimated_AQI",
                title="Hourly project-specific estimated AQI proxy",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No complete hourly AQI values available.")
    else:
        st.info("Hourly processed data is not available yet.")

    st.caption(
        "Hourly estimated AQI is a four-pollutant project proxy, "
        "not an official CPCB AQI measurement."
    )

    # Hourly data quality
    st.subheader("Hourly Data Quality")

    missing_counts = hourly_quality.get(
        "missing_values_by_pollutant", {}
    )

    if missing_counts:
        total = max(total_hours, 1)

        quality_table = pd.DataFrame([
            {
                "Pollutant": pollutant,
                "Missing hours": count,
                "Missing percentage": round(
                    int(count) / total * 100, 2
                )
            }
            for pollutant, count in missing_counts.items()
        ])

        st.dataframe(
            quality_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Hourly quality report is not available.")

    st.divider()

    # Pollution episodes
    st.subheader("Detected Pollution Episodes")

    episodes = hourly_events.get("episodes", [])

    if episodes:
        episode_df = pd.DataFrame(episodes)
        st.dataframe(
            episode_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No pollution episodes reported.")

    # Individual anomaly alerts
    st.subheader("Hourly Anomaly Alerts")

    if not hourly_anomalies.empty:
        alert_df = hourly_anomalies[
            hourly_anomalies["status"] == "Potential pollution spike"
        ].copy()

        if not alert_df.empty:
            alert_df["period_start"] = pd.to_datetime(
                alert_df["period_start"], errors="coerce"
            )

            alert_df = alert_df.sort_values(
                "period_start", ascending=False
            )

            st.dataframe(
                alert_df.head(30),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No potential spike alerts in the latest report.")
    else:
        st.info("Anomaly report is not available.")

    st.divider()

    # Retraining decision signals
    st.subheader("Intelligent Self-Retraining Decision")

    signals = adaptive_decision.get("monitoring_signals", {})
    readiness = adaptive_decision.get("training_readiness", {})

    a, b, c, d = st.columns(4)

    a.metric(
        "Confirmed drift",
        "Yes" if signals.get("confirmed_drift") else "No"
    )

    b.metric(
        "Performance assessed",
        "Yes" if signals.get("performance_assessed") else "No"
    )

    c.metric(
        "Performance degraded",
        "Yes" if signals.get("performance_degraded") else "No"
    )

    d.metric(
        "Evaluated predictions",
        signals.get("evaluated_predictions", 0)
    )

    available = readiness.get("available_labelled_rows", 0)
    minimum = readiness.get("minimum_required_rows", 90)

    st.progress(
        min(float(available) / max(float(minimum), 1), 1.0),
        text=f"Labelled training data: {available} / {minimum}"
    )

    st.write("**Current decision:**", decision_name)
    st.write("**Reason:**", adaptive_decision.get("reason", "Not available"))

    if adaptive_decision:
        with st.expander("View complete decision report"):
            st.json(adaptive_decision)

    st.info(
        "A pollution anomaly alone does not trigger retraining. "
        "Event-aware forecasting and automatic model deployment are "
        "separate capabilities still under development."
    )


# ---------------- MLFLOW EXPERIMENT TRACKING ----------------

elif page == "MLflow Experiment Tracking":

    st.title("MLflow Experiment Tracking")
    st.caption("Experiment history, model comparison and training metrics.")

    if not MLFLOW_DB.exists():
        st.error("MLflow database not found.")
        st.info(f"Expected database location: {MLFLOW_DB}")
    else:
        try:
            mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

            experiment = mlflow.get_experiment_by_name(
                "AeroAI-AQI-Model-Training"
            )

            if experiment is None:
                st.warning("AeroAI-AQI-Model-Training experiment not found.")
            else:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id]
                )

                st.subheader("Experiment overview")

                total_runs = len(runs)
                finished_runs = int((runs["status"] == "FINISHED").sum())
                failed_runs = int((runs["status"] == "FAILED").sum())

                a, b, c = st.columns(3)
                a.metric("Total runs", total_runs)
                b.metric("Successful runs", finished_runs)
                c.metric("Failed runs", failed_runs)

                st.caption(
                    f"Experiment: {experiment.name} | "
                    f"Experiment ID: {experiment.experiment_id}"
                )

                if not runs.empty:
                    run_name_col = "tags.mlflow.runName"

                    if run_name_col in runs.columns:
                        model_runs = runs[
                            runs[run_name_col].isin(["Random Forest", "XGBoost"])
                            & (runs["status"] == "FINISHED")
                        ].copy()

                        if "end_time" in model_runs.columns:
                            model_runs = model_runs.sort_values(
                                "end_time", ascending=False
                            )

                        # Keep the latest successful run for each model
                        if not model_runs.empty:
                            latest_models = model_runs.drop_duplicates(
                                subset=[run_name_col]
                            ).copy()

                            st.subheader("Latest successful model runs")

                            display_columns = [run_name_col]
                            for metric_name in ["MAE", "RMSE", "R2"]:
                                col = f"metrics.{metric_name}"
                                if col in latest_models.columns:
                                    display_columns.append(col)

                            st.dataframe(
                                latest_models[display_columns].rename(
                                    columns={
                                        run_name_col: "Model",
                                        "metrics.MAE": "MAE",
                                        "metrics.RMSE": "RMSE",
                                        "metrics.R2": "R²"
                                    }
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.subheader("Model comparison")

                            available_metrics = [
                                m for m in ["MAE", "RMSE", "R2"]
                                if f"metrics.{m}" in latest_models.columns
                            ]

                            if available_metrics:
                                selected_metric = st.selectbox(
                                    "Select evaluation metric",
                                    available_metrics,
                                    format_func=lambda x: "R²" if x == "R2" else x
                                )

                                chart_data = latest_models[
                                    [run_name_col, f"metrics.{selected_metric}"]
                                ].copy()

                                chart_data[f"metrics.{selected_metric}"] = pd.to_numeric(
                                    chart_data[f"metrics.{selected_metric}"],
                                    errors="coerce"
                                )

                                fig = px.bar(
                                    chart_data,
                                    x=run_name_col,
                                    y=f"metrics.{selected_metric}",
                                    color=run_name_col,
                                    title=f"Model comparison — {selected_metric}"
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    showlegend=False,
                                    xaxis_title="Model",
                                    yaxis_title=selected_metric
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            best_row = latest_models[
                                latest_models[run_name_col] == "XGBoost"
                            ]

                            if not best_row.empty:
                                st.success(
                                    "XGBoost is the best model in the recorded "
                                    "historical comparison."
                                )
                                row = best_row.iloc[0]
                                x, y, z = st.columns(3)

                                x.metric(
                                    "XGBoost MAE",
                                    f"{row['metrics.MAE']:.3f}"
                                    if pd.notna(row.get("metrics.MAE"))
                                    else "N/A"
                                )
                                y.metric(
                                    "XGBoost RMSE",
                                    f"{row['metrics.RMSE']:.3f}"
                                    if pd.notna(row.get("metrics.RMSE"))
                                    else "N/A"
                                )
                                z.metric(
                                    "XGBoost R²",
                                    f"{row['metrics.R2']:.3f}"
                                    if pd.notna(row.get("metrics.R2"))
                                    else "N/A"
                                )
                        else:
                            st.info("No successful Random Forest or XGBoost model runs found.")

                    # Summary run metrics
                    summary_runs = runs[
                        runs[run_name_col].astype(str).str.contains(
                            "Summary", case=False, na=False
                        )
                        & (runs["status"] == "FINISHED")
                    ] if run_name_col in runs.columns else pd.DataFrame()

                    if not summary_runs.empty:
                        st.subheader("Best-model summary run")
                        summary = summary_runs.iloc[0]
                        summary_metrics = {
                            key.replace("metrics.", ""): value
                            for key, value in summary.items()
                            if key.startswith("metrics.")
                            and pd.notna(value)
                        }
                        if summary_metrics:
                            st.json(summary_metrics)
                        else:
                            st.caption(
                                "The summary run exists, but no summary metrics "
                                "were found under the metrics fields."
                            )

                    st.subheader("Complete experiment run history")

                    history_columns = [
                        c for c in [
                            "tags.mlflow.runName",
                            "status",
                            "start_time",
                            "end_time",
                            "metrics.MAE",
                            "metrics.RMSE",
                            "metrics.R2"
                        ]
                        if c in runs.columns
                    ]

                    st.dataframe(
                        runs[history_columns].rename(
                            columns={
                                "tags.mlflow.runName": "Run name",
                                "metrics.MAE": "MAE",
                                "metrics.RMSE": "RMSE",
                                "metrics.R2": "R²"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.info("No MLflow runs are recorded in this experiment.")

                st.info(
                    "Historical metrics are from the 2024 chronological test set. "
                    "They do not represent verified performance on newer API observations."
                )

        except Exception as e:
            st.error(f"Unable to load MLflow experiment data: {e}")
            st.caption(
                "Check that MLflow is installed and mlflow.db is accessible."
            )

# ---------------- PIPELINE STATUS ----------------
elif page == "Pipeline Status":
    st.title("Pipeline Status")
    st.caption("Local project artifacts and pipeline execution controls.")

    artifact_rows = [
        ("Historical AQI dataset", HISTORICAL_FILE),
        ("Active trained model", MODEL_FILE),
        ("Model evaluation results", RESULTS_FILE),
        ("Latest API measurements (processed)", LATEST_FILE),
        ("Daily preprocessed data", DAILY_FILE),
        ("Prediction history", PREDICTION_FILE),
        ("Prediction evaluation", EVALUATION_FILE),
        ("Model monitoring summary", MONITORING_FILE),
        ("Data drift report", DRIFT_FILE),
        ("Retraining decision report", RETRAINING_FILE),
        ("Adaptive retraining decision", RETRAINING_DECISION_FILE),
        ("Hourly raw dataset", HOURLY_RAW_FILE),
        ("Hourly processed dataset", HOURLY_PROCESSED_FILE),
        ("Hourly data quality report", HOURLY_QUALITY_FILE),
        ("Hourly anomaly report", HOURLY_ANOMALY_FILE),
        ("Pollution episode report", HOURLY_EVENTS_FILE),
        ("Pipeline runner", PIPELINE_SCRIPT)
    ]
    for name, path in artifact_rows:
        exists = path.exists()
        left, right = st.columns([2, 3])
        left.write(name)
        if exists:
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y %H:%M:%S")
            right.success(f"Available · {path.relative_to(ROOT)} · Updated {modified}")
        else:
            right.warning(f"Missing · {path.relative_to(ROOT)}")

    st.subheader("Run automated pipeline")
    st.write("Runs collection, preprocessing, AQI calculation, prediction, drift monitoring, performance monitoring and retraining checks in sequence.")
    if st.button("▶ Run pipeline now", type="primary"):
        with st.spinner("AeroAI pipeline is running. This may take a few moments..."):
            success, output = run_pipeline()
        st.session_state["pipeline_output"] = output
        st.session_state["pipeline_success"] = success
        st.cache_data.clear()
        if success:
            st.success("Pipeline completed successfully.")
        else:
            st.error("Pipeline failed. Review the output below.")

    if "pipeline_output" in st.session_state:
        st.subheader("Latest run output")
        st.code(st.session_state["pipeline_output"], language="text")

    st.subheader("Pipeline architecture")
    st.code(
        "XKDR API data collection\n"
        "        ↓\n"
        "Preprocessing & validation\n"
        "        ↓\n"
        "Partial-pollutant AQI estimation\n"
        "        ↓\n"
        "Next-day XGBoost prediction\n"
        "        ↓\n"
        "PSI data drift monitoring\n"
        "        ↓\n"
        "Prediction outcome monitoring\n"
        "        ↓\n"
        "Guarded candidate retraining",
        language="text"
    )

    st.info(
        "The pipeline runner is currently manually triggered from this page or the terminal. "
        "Daily scheduling and MLflow experiment tracking are not yet configured."
    )

st.sidebar.divider()
st.sidebar.caption("AeroAI • Academic project prototype")
st.sidebar.caption("Estimated AQI — not official CPCB AQI")
