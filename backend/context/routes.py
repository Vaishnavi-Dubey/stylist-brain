"""
routes.py — Context endpoints (weather + calendar)

GET /context/weather            — current weather (OpenWeatherMap free tier)
GET /context/calendar           — today's first calendar event (Google OAuth)
"""

import logging

from fastapi import APIRouter, Query
from context.weather import get_weather, DEFAULT_CITY
from context.calendar import get_todays_event

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/weather", summary="Current weather for styling context")
def weather(city: str = Query(default="", description="Override city name")):
    """
    Fetch current weather from OpenWeatherMap free tier.

    Set OWM_API_KEY in your .env file.
    Falls back to safe defaults (22°C, Unknown) if the key is missing.
    """
    return get_weather(city.strip() or DEFAULT_CITY)


@router.get("/calendar", summary="Today's first calendar event")
def calendar():
    """
    Return the first Google Calendar event today as a plain-text string.

    Requires credentials.json in the project root (Google OAuth — free).
    Returns null event if not configured or no events today.
    """
    event = get_todays_event()
    return {"event": event}
