"""Helpers for transforming cached daily bars into completed weekly bars."""

from __future__ import annotations

from datetime import datetime, timedelta

import baostock as bs
import pandas as pd


def get_last_trading_day_of_week(target_date: pd.Timestamp) -> pd.Timestamp:
    """返回指定日期所在周的最后一个交易日。

    规则：
    1. 查找本周一到周五的交易日历
    2. 返回本周最后一个交易日（可能是周五、周四等）
    3. 如果本周没有交易日（极端情况），返回上周五
    """
    # 计算本周的周一和周五
    weekday = target_date.weekday()

    # 计算本周的周一（可能在上周）
    days_to_monday = 0 - weekday
    if days_to_monday > 0:
        days_to_monday = days_to_monday - 7
    this_week_monday = target_date + pd.Timedelta(days=days_to_monday)

    # 计算本周的周五（可能在下周）
    days_to_friday = (4 - weekday) % 7
    this_week_friday = target_date + pd.Timedelta(days=days_to_friday)

    # 查询本周一到周五的交易日历
    start_date = this_week_monday.strftime("%Y-%m-%d")
    end_date = this_week_friday.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            # 如果查询失败，保守返回上周五
            return this_week_friday - pd.Timedelta(days=7)

        trading_days = []
        while rs.next():
            calendar_date, is_trading_day = rs.get_row_data()
            if is_trading_day == "1":
                trading_days.append(pd.Timestamp(calendar_date))

        if trading_days:
            # 返回本周最后一个交易日
            return max(trading_days)
        else:
            # 本周没有交易日（极端情况），返回上周五
            return this_week_friday - pd.Timedelta(days=7)
    except Exception:
        # 如果baostock调用失败，保守返回上周五
        return this_week_friday - pd.Timedelta(days=7)


def ensure_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily bar data before strategy calculations."""
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values("date").reset_index(drop=True)

    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])


def get_last_completed_week_end(now: datetime | None = None, daily_latest_date: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the Friday date for the latest fully completed trading week.

    Rules:
    1. Monday-Thursday: always return previous Friday
    2. Friday before 20:00: return previous Friday
    3. Friday after 20:00: if this week's data is complete, return this Friday; otherwise return previous Friday
    4. Saturday/Sunday: if this week's data is complete, return this Friday; otherwise return previous Friday

    This week's data is considered complete if daily_latest_date is on or after
    the last trading day of this week (which could be a Thursday for holiday-shortened weeks).
    """
    if now is None:
        now = datetime.now()

    current_date = pd.Timestamp(now.date())
    weekday = current_date.weekday()
    hour = now.hour

    # Apply rules
    if weekday < 4:  # Monday-Thursday
        # Always previous Friday
        days_since_friday = weekday + 3
    elif weekday == 4:  # Friday
        if hour >= 20:
            # Friday after 20:00
            if daily_latest_date is not None:
                # 获取本周最后一个交易日
                last_trading_day_of_week = get_last_trading_day_of_week(current_date)
                # 检查是否已经有本周最后一个交易日的数据
                if daily_latest_date >= last_trading_day_of_week:
                    return current_date  # This Friday
                else:
                    days_since_friday = 7  # Previous Friday
            else:
                days_since_friday = 7  # Previous Friday
        else:
            # Friday before 20:00
            days_since_friday = 7  # Previous Friday (last week's)
    else:  # Saturday or Sunday (weekday >= 5)
        if daily_latest_date is not None:
            # 获取本周最后一个交易日
            last_trading_day_of_week = get_last_trading_day_of_week(current_date)
            # 检查是否已经有本周最后一个交易日的数据
            if daily_latest_date >= last_trading_day_of_week:
                days_since_friday = weekday - 4  # This Friday
            else:
                days_since_friday = weekday + 3  # Previous Friday
        else:
            days_since_friday = weekday + 3  # Previous Friday

    result = current_date - pd.Timedelta(days=days_since_friday)
    return result


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
    """Return weekly bars while ignoring the currently open week.

    The cutoff date is determined by both current time and the latest daily data available.
    """
    # Get the latest date from the daily data
    daily = ensure_daily_frame(df)
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    daily_latest_date = pd.to_datetime(daily["date"]).max()

    # Determine cutoff date based on current time and available daily data
    cutoff_date = get_last_completed_week_end(now, daily_latest_date)
    return convert_daily_to_weekly(df, cutoff_date=cutoff_date)
