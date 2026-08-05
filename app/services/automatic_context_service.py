from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


class AutomaticContextService:
    """Keyless online weather context with strict freshness handling."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, refresh_seconds: int = 900, stale_seconds: int = 1800) -> None:
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.stale_seconds = max(self.refresh_seconds, int(stale_seconds))
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "GuardianOS/6.3 automatic-context"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _weather_label(code: int, precipitation: float, snowfall: float) -> str:
        if snowfall > 0 or code in {71, 73, 75, 77, 85, 86}:
            return "snow"
        if precipitation > 0 or code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}:
            return "rain"
        if code in {45, 48}:
            return "fog"
        if code in {0, 1}:
            return "clear"
        if code in {2, 3}:
            return "cloudy"
        return "unknown"

    @staticmethod
    def _road_condition(weather: str, temperature: float, precipitation: float) -> str:
        if weather == "snow" or (temperature <= 1.0 and precipitation > 0):
            return "snow_or_ice"
        if weather == "rain" or precipitation > 0:
            return "wet"
        if weather in {"clear", "cloudy", "fog"}:
            return "dry"
        return "unknown"

    def _geocode(self, location: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
        payload = self._get_json(f"{self.GEOCODING_URL}?{query}")
        results = payload.get("results") or []
        if not results:
            raise ValueError(f'Location “{location}” was not found.')
        result = results[0]
        return {
            "latitude": float(result["latitude"]),
            "longitude": float(result["longitude"]),
            "name": str(result.get("name") or location),
            "admin1": str(result.get("admin1") or ""),
            "country": str(result.get("country") or ""),
            "timezone": str(result.get("timezone") or "auto"),
        }

    def _fetch(self, location: str) -> dict[str, Any]:
        geo = self._geocode(location)
        current_fields = (
            "temperature_2m,apparent_temperature,precipitation,rain,showers,"
            "snowfall,weather_code,cloud_cover,is_day,wind_speed_10m,visibility"
        )
        query = urllib.parse.urlencode({
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "current": current_fields,
            "timezone": "auto",
            "forecast_days": 1,
        })
        payload = self._get_json(f"{self.FORECAST_URL}?{query}")
        current = payload.get("current") or {}
        precipitation = float(current.get("precipitation", 0) or 0)
        snowfall = float(current.get("snowfall", 0) or 0)
        temperature = float(current.get("temperature_2m", 0) or 0)
        raw_weather_code = current.get("weather_code")
        weather_code = -1 if raw_weather_code is None else int(raw_weather_code)
        weather = self._weather_label(weather_code, precipitation, snowfall)
        is_day = int(current.get("is_day", 1) or 0)
        external_light = "daylight" if is_day else "night"
        updated_at = self._now()
        display_location = ", ".join(filter(None, [geo["name"], geo["admin1"], geo["country"]]))
        return {
            "available": True,
            "online": True,
            "location_query": location,
            "location": display_location,
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "timezone": payload.get("timezone") or geo["timezone"],
            "weather": weather,
            "road_condition": self._road_condition(weather, temperature, precipitation),
            "external_light": external_light,
            "temperature_c": round(temperature, 1),
            "apparent_temperature_c": round(float(current.get("apparent_temperature", temperature) or temperature), 1),
            "precipitation_mm": round(precipitation, 2),
            "cloud_cover_percent": round(float(current.get("cloud_cover", 0) or 0), 1),
            "wind_speed_kmh": round(float(current.get("wind_speed_10m", 0) or 0), 1),
            "visibility_m": round(float(current.get("visibility", 0) or 0), 0),
            "weather_code": weather_code,
            "observed_at": current.get("time"),
            "updated_at": updated_at,
            "source": "Open-Meteo",
            "confidence": 0.95,
            "fresh": True,
            "error": None,
        }

    def snapshot(self, location: str, force: bool = False) -> dict[str, Any]:
        clean = str(location or "").strip()
        if not clean:
            return self.unknown("Set a city or locality to enable automatic weather.")
        key = clean.casefold()
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached and not force and now - float(cached.get("fetched_epoch", 0)) < self.refresh_seconds:
                return self._freshness(dict(cached["payload"]), now - cached["fetched_epoch"])
        try:
            payload = self._fetch(clean)
            with self._lock:
                self._cache[key] = {"fetched_epoch": now, "payload": dict(payload)}
            return payload
        except Exception as error:
            # Never present old weather as current. A cached result may be shown
            # only while still within the strict stale window and is labelled.
            with self._lock:
                cached = self._cache.get(key)
            if cached:
                age = now - float(cached.get("fetched_epoch", 0))
                if age <= self.stale_seconds:
                    payload = self._freshness(dict(cached["payload"]), age)
                    payload.update({"online": False, "fresh": False, "error": f"Refresh failed: {type(error).__name__}"})
                    return payload
            return self.unknown(f"Automatic weather unavailable: {type(error).__name__}.", location=clean)

    def _freshness(self, payload: dict[str, Any], age_seconds: float) -> dict[str, Any]:
        payload["age_seconds"] = round(max(0.0, age_seconds))
        payload["fresh"] = age_seconds <= self.refresh_seconds
        return payload

    def unknown(self, message: str, location: str = "") -> dict[str, Any]:
        return {
            "available": False, "online": False, "location_query": location,
            "location": location or "Not configured", "weather": "unknown",
            "road_condition": "unknown", "external_light": "unknown",
            "temperature_c": None, "apparent_temperature_c": None,
            "precipitation_mm": None, "cloud_cover_percent": None,
            "wind_speed_kmh": None, "visibility_m": None,
            "weather_code": None, "observed_at": None, "updated_at": None,
            "source": "Unavailable", "confidence": 0.0, "fresh": False,
            "age_seconds": None, "error": message,
        }
