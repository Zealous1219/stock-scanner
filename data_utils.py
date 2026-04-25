"""Helpers for transforming cached daily bars into completed weekly bars."""

from __future__ import annotations

import logging
from datetime import datetime

import baostock as bs
import pandas as pd

logger = logging.getLogger(__name__)


def get_week_monday_friday(target_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """返回指定日期所在自然周的周一和周五。

    使用固定的周五锚点（W-FRI），即使最后交易日是周四，
    完成周线的锚点也仍然应该是该周对应的周五日期。
    """
    # 周一：减去 weekday 天数（周一 weekday=0，所以不变）
    this_week_monday = target_date - pd.Timedelta(days=target_date.weekday())
    # 周五：周一 + 4 天
    this_week_friday = this_week_monday + pd.Timedelta(days=4)
    return this_week_monday, this_week_friday


def get_last_trading_day_of_week(target_date: pd.Timestamp) -> pd.Timestamp | None:
    """返回指定日期所在周的最后一个交易日。

    规则：
    1. 查找本周一到周五的交易日历
    2. 返回本周最后一个交易日（可能是周五、周四等）
    3. 如果本周没有交易日，返回 None（无法确认完成状态）
    4. 如果查询失败，返回 None（不能误判本周为完成）

    Returns:
        pd.Timestamp: 本周最后一个交易日
        None: 查询失败或无法确认本周完成状态
    """
    this_week_monday, this_week_friday = get_week_monday_friday(target_date)

    start_date = this_week_monday.strftime("%Y-%m-%d")
    end_date = this_week_friday.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            # 查询失败，明确记录日志并返回 None
            logger.error(
                "交易日历查询失败 - target_date=%s, query_range=[%s, %s], error_code=%s, error_msg=%s",
                target_date.date(),
                start_date,
                end_date,
                rs.error_code,
                rs.error_msg,
            )
            return None

        trading_days = []
        while rs.next():
            calendar_date, is_trading_day = rs.get_row_data()
            if is_trading_day == "1":
                trading_days.append(pd.Timestamp(calendar_date))

        if trading_days:
            # 返回本周最后一个交易日
            return max(trading_days)
        else:
            # 本周没有交易日，无法确认完成状态
            logger.warning(
                "本周没有交易日 - target_date=%s, query_range=[%s, %s], 无法确认完成状态",
                target_date.date(),
                start_date,
                end_date,
            )
            return None
    except Exception as exc:
        # baostock调用失败，明确记录日志并返回 None
        logger.error(
            "交易日历查询异常 - target_date=%s, query_range=[%s, %s], exception=%s",
            target_date.date(),
            start_date,
            end_date,
            exc,
        )
        return None


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
    last trading day of this week (which could be a Thursday for holiday-shortened weeks).

    Returns a Friday anchor point (for W-FRI resample), even if last trading day was Thursday.
    """
    if now is None:
        now = datetime.now()

    current_date = pd.Timestamp(now.date())
    weekday = current_date.weekday()
    hour = now.hour

    # Get the Friday anchor point for the week containing current_date
    _, this_friday = get_week_monday_friday(current_date)
    prev_friday = this_friday - pd.Timedelta(days=7)

    # Monday-Thursday: always return previous Friday
    if weekday < 4:
        logger.debug(
            "周线完成判定 - now=%s, weekday=%s, 周一到周四回退上周五: %s",
            now,
            weekday,
            prev_friday.date(),
        )
        return prev_friday

    # Friday
    if weekday == 4:
        # Before 20:00: return previous Friday
        if hour < 20:
            logger.debug(
                "周线完成判定 - now=%s, 周五 20:00 前回退上周五: %s",
                now,
                prev_friday.date(),
            )
            return prev_friday

        # After 20:00: check if this week's data is complete
        if daily_latest_date is not None:
            last_trading_day = get_last_trading_day_of_week(current_date)
            if last_trading_day is None:
                # 查询失败，保守回退到上周五
                logger.warning(
                    "周线完成判定 - now=%s, daily_latest_date=%s, 交易日历查询失败，保守回退到上周五: %s",
                    now,
                    daily_latest_date.date(),
                    prev_friday.date(),
                )
                return prev_friday
            if daily_latest_date >= last_trading_day:
                logger.debug(
                    "周线完成判定 - now=%s, daily_latest_date=%s, 本周最后交易日=%s, 返回本周五锚点: %s",
                    now,
                    daily_latest_date.date(),
                    last_trading_day.date(),
                    this_friday.date(),
                )
                return this_friday  # This week is complete
            else:
                logger.debug(
                    "周线完成判定 - now=%s, daily_latest_date=%s, 本周最后交易日=%s, 数据未到达，回退上周五: %s",
                    now,
                    daily_latest_date.date(),
                    last_trading_day.date(),
                    prev_friday.date(),
                )
                return prev_friday  # This week not complete
        else:
            logger.warning(
                "周线完成判定 - now=%s, daily_latest_date 为 None，回退上周五: %s",
                now,
                prev_friday.date(),
            )
            return prev_friday  # No daily data

    # Saturday or Sunday (weekday >= 5)
    # Check if this week's data is complete
    if daily_latest_date is not None:
        last_trading_day = get_last_trading_day_of_week(current_date)
        if last_trading_day is None:
            # 查询失败，保守回退到上周五
            logger.warning(
                "周线完成判定 - now=%s, daily_latest_date=%s, 交易日历查询失败，保守回退到上周五: %s",
                now,
                daily_latest_date.date(),
                prev_friday.date(),
            )
            return prev_friday
        if daily_latest_date >= last_trading_day:
            logger.debug(
                "周线完成判定 - now=%s, daily_latest_date=%s, 本周最后交易日=%s, 返回本周五锚点: %s",
                now,
                daily_latest_date.date(),
                last_trading_day.date(),
                this_friday.date(),
            )
            return this_friday  # This week is complete
        else:
            logger.debug(
                "周线完成判定 - now=%s, daily_latest_date=%s, 本周最后交易日=%s, 数据未到达，回退上周五: %s",
                now,
                daily_latest_date.date(),
                last_trading_day.date(),
                prev_friday.date(),
            )
            return prev_friday  # This week not complete
    else:
        logger.warning(
            "周线完成判定 - now=%s, daily_latest_date 为 None，回退上周五: %s",
            now,
            prev_friday.date(),
        )
        return prev_friday  # No daily data


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
