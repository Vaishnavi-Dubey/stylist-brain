"""
weather.py — OpenWeatherMap free-tier integration
Fetches today's weather and injects it into the Ollama styling prompt.

Free tier: no credit card, 60 calls/min, 1M calls/month.
Sign up at https://openweathermap.org/api → "Current Weather Data" (free plan).
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
# Set OWM_API_KEY in your environment or a .env file — never hardcode it.
OWM_API_KEY   = os.getenv("OWM_API_KEY", "")
DEFAULT_CITY  = os.getenv("WEATHER_CITY", "Mumbai")   # change to your city


def get_weather(city: str = DEFAULT_CITY) -> dict:
    """
    Fetch current weather for *city* from OpenWeatherMap free API.

    Args:
        city: City name string, e.g. "London" or "New York".

    Returns:
        Dict with keys: city, temp_c, feels_like_c, condition, humidity_pct.
        On error returns a safe fallback dict so the app never crashes.
    """
    if not OWM_API_KEY:
        logger.warning("OWM_API_KEY not set — returning fallback weather")
        return _fallback(city)

    try:
        resp = httpx.get(
            OWM_BASE_URL,
            params={
                "q":     city,
                "appid": OWM_API_KEY,
                "units": "metric",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "city":         data["name"],
            "temp_c":       round(data["main"]["temp"], 1),
            "feels_like_c": round(data["main"]["feels_like"], 1),
            "condition":    data["weather"][0]["description"].capitalize(),
            "humidity_pct": data["main"]["humidity"],
        }

    except (httpx.HTTPError, KeyError) as exc:
        logger.error("Weather fetch failed: %s", exc)
        return _fallback(city)


def _fallback(city: str) -> dict:
    return {
        "city":         city,
        "temp_c":       22.0,
        "feels_like_c": 22.0,
        "condition":    "Unknown",
        "humidity_pct": 50,
    }
