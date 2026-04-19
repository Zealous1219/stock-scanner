"""Helpers for transforming cached daily bars into completed weekly bars."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def ensure_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily bar data before strategy calculations."""
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values("date").reset_index(drop=True)

    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])


def get_last_completed_week_end(now: datetime | None = None) -> pd.Timestamp:
    """Return the Friday date for the latest fully completed trading week."""
    if now is None:
        now = datetime.now()

    current_date = pd.Timestamp(now.date())
    weekday = current_date.weekday()

    if weekday >= 5:
        days_since_friday = weekday - 4
    else:
        days_since_friday = weekday + 3

    return current_date - pd.Timedelta(days=days_since_friday)


def convert_daily_to_weekly(df: pd.DataFrame, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate daily bars into completed Friday-anchored weekly bars."""
    daily = ensure_daily_frame(df)
    if cutoff_date is not None:
        daily = daily[daily["date"] <= cutoff_date]

    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    weekly = (
        daily.set_index("date")
        .resample("W-FRI")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    return weekly


def get_completed_weekly_bars(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Return weekly bars while ignoring the currently open week."""
    cutoff_date = get_last_completed_week_end(now)
    return convert_daily_to_weekly(df, cutoff_date=cutoff_date)
