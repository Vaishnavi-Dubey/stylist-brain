"""
calendar.py — Google Calendar OAuth integration
Reads today's first calendar event and returns it as a plain string
for injection into the Ollama styling prompt.

Uses Google Calendar API (free, no billing required).
OAuth credentials: https://console.cloud.google.com → APIs & Services → Credentials
Download credentials.json and place it at the project root.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path(__file__).parents[3] / "credentials.json"
TOKEN_FILE       = Path(__file__).parents[3] / "token.json"
SCOPES           = ["https://www.googleapis.com/auth/calendar.readonly"]


def get_todays_event() -> str | None:
    """
    Return a plain-text description of the first Google Calendar event today.

    Returns:
        Event description string, or None if no events / not configured.

    Example return:
        "Team standup at 10:00 AM (30 min)"
    """
    if not CREDENTIALS_FILE.exists():
        logger.info("credentials.json not found — skipping calendar context")
        return None

    try:
        from google.oauth2.credentials import Credentials          # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow     # type: ignore
        from google.auth.transport.requests import Request         # type: ignore
        from googleapiclient.discovery import build                # type: ignore
    except ImportError:
        logger.warning(
            "Google API client not installed. "
            "Run: pip install google-api-python-client google-auth-oauthlib"
        )
        return None

    creds = _get_or_refresh_credentials(Credentials, InstalledAppFlow, Request)
    if creds is None:
        return None

    service = build("calendar", "v3", credentials=creds)

    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0).isoformat()
    end_of_day   = now.replace(hour=23, minute=59, second=59).isoformat()

    try:
        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day,
            timeMax=end_of_day,
            maxResults=1,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = events_result.get("items", [])
        if not events:
            return None

        event = events[0]
        summary = event.get("summary", "Untitled event")
        start   = event.get("start", {}).get("dateTime", "")
        return f"{summary} at {_fmt_time(start)}" if start else summary

    except Exception as exc:  # noqa: BLE001
        logger.error("Calendar API error: %s", exc)
        return None


# ── Internal helpers ────────────────────────────────────────────────────────────

def _get_or_refresh_credentials(Credentials, InstalledAppFlow, Request):
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token refresh failed: %s", exc)

    # First-time OAuth flow — opens browser once, saves token for future runs
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_token(creds)
    return creds


def _save_token(creds) -> None:
    TOKEN_FILE.write_text(creds.to_json())


def _fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%I:%M %p")
    except ValueError:
        return iso
