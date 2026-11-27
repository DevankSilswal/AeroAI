"""Dehradun weather integration using Open-Meteo."""

import json
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "incoming" / "weather_snapshot.json"

LATITUDE = 30.269444
LONGITUDE = 78.044167

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    "&current=temperature_2m,relative_humidity_2m,"
    "apparent_temperature,precipitation,weather_code,"
    "cloud_cover,wind_speed_10m,wind_direction_10m"
    "&daily=uv_index_max,temperature_2m_max,temperature_2m_min"
    "&timezone=Asia%2FKolkata&forecast_days=1"
)


def fetch_weather():
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "AeroAI-Academic-Project/1.0"}
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"Weather API returned HTTP {response.status}")
        data = json.loads(response.read().decode("utf-8"))

    current = data.get("current", {})
    daily = data.get("daily", {})

    required = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "weather_code",
        "wind_speed_10m",
    ]

    missing = [key for key in required if current.get(key) is None]
    if missing:
        raise ValueError(f"Weather response missing fields: {missing}")

    snapshot = {
        "location": "Doon University, Dehradun",
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "current": current,
        "daily": {
            "date": daily.get("time", [None])[0],
            "uv_index_max": daily.get("uv_index_max", [None])[0],
            "temperature_max": daily.get("temperature_2m_max", [None])[0],
            "temperature_min": daily.get("temperature_2m_min", [None])[0],
        },
        "fetched_at": datetime.now().astimezone().isoformat(),
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "status": "success",
    }

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = CACHE_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(snapshot, indent=2))
    temporary_file.replace(CACHE_FILE)

    return snapshot


def get_weather():
    """Fetch fresh weather; use cached data if the API is unavailable."""
    try:
        return fetch_weather()
    except Exception as exc:
        if CACHE_FILE.exists():
            try:
                cached = json.loads(CACHE_FILE.read_text())
                cached["status"] = "cached"
                cached["error"] = str(exc)
                return cached
            except Exception:
                pass

        return {
            "status": "unavailable",
            "error": str(exc),
            "source": "Open-Meteo",
        }


def weather_description(code):
    """Convert WMO weather codes into readable descriptions."""
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail",
    }
    try:
        return descriptions.get(int(code), "Unknown conditions")
    except (TypeError, ValueError):
        return "Unknown conditions"


if __name__ == "__main__":
    result = get_weather()
    print(json.dumps(result, indent=2))
