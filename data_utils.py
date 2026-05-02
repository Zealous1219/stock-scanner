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


def get_snapshot_trading_week_info(snapshot_date: datetime | pd.Timestamp) -> dict:
    """返回 snapshot 对应周的交易日历元数据。

    该函数用于校验历史 snapshot 是否为有效交易周，解决语义可解释性问题：
    - 当前 generate_weekly_snapshot_dates() 按自然周（每7天）生成历史 snapshot
    - 若某历史自然周整周休市，仍会生成该周 snapshot
    - 本函数提供元数据，让 replay/分析时能区分“真实有交易完成的周”与“日历上凑出来的周锚点”

    参数：
        snapshot_date: 快照日期（通常是周五锚点）

    返回字典包含以下字段：
        - snapshot_date: 输入的快照日期（pd.Timestamp）
        - friday_anchor: 同 snapshot_date，明确为周五锚点
        - week_start_date: 本周周一日期
        - week_end_date: 本周周五日期
        - last_trading_day_of_week: 本周最后一个交易日（如果有）
        - has_trading_day: 本周是否有至少一个交易日
        - is_valid_completed_trading_week: 是否为有效完成交易周
        - calendar_query_ok: 交易日历查询是否成功
        - trading_days_count: 本周交易日数量

    语义说明：
        - has_trading_day=True 且 calendar_query_ok=True: 本周有实际交易
        - has_trading_day=False 且 calendar_query_ok=True: 本周整周无交易（如节假日）
        - calendar_query_ok=False: 交易日历查询失败，状态未知
        - is_valid_completed_trading_week: 对历史 snapshot，当 has_trading_day=True 时为 True
    """
    # 统一转换为 pd.Timestamp 并归一化到日期级（去掉时间分量）
    snap_ts = pd.Timestamp(snapshot_date).normalize()

    # 获取本周周一和周五
    week_start_date, week_end_date = get_week_monday_friday(snap_ts)

    # 初始化返回结果
    result = {
        "snapshot_date": snap_ts,
        "friday_anchor": snap_ts,
        "week_start_date": week_start_date,
        "week_end_date": week_end_date,
        "last_trading_day_of_week": None,
        "has_trading_day": False,
        "is_valid_completed_trading_week": False,
        "calendar_query_ok": False,
        "trading_days_count": 0,
    }

    # 查询交易日历
    start_date = week_start_date.strftime("%Y-%m-%d")
    end_date = week_end_date.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            logger.error(
                "snapshot交易日历查询失败 - snapshot_date=%s, query_range=[%s, %s], error_code=%s, error_msg=%s",
                snap_ts.date(),
                start_date,
                end_date,
                rs.error_code,
                rs.error_msg,
            )
            return result  # calendar_query_ok 保持 False

        trading_days = []
        while rs.next():
            calendar_date, is_trading_day = rs.get_row_data()
            if is_trading_day == "1":
                trading_days.append(pd.Timestamp(calendar_date))

        result["calendar_query_ok"] = True
        result["trading_days_count"] = len(trading_days)

        if trading_days:
            result["has_trading_day"] = True
            result["last_trading_day_of_week"] = max(trading_days)
            result["is_valid_completed_trading_week"] = True
        else:
            # 本周没有交易日（如整周休市）
            result["has_trading_day"] = False
            result["is_valid_completed_trading_week"] = False

    except Exception as exc:
        logger.error(
            "snapshot交易日历查询异常 - snapshot_date=%s, query_range=[%s, %s], exception=%s",
            snap_ts.date(),
            start_date,
            end_date,
            exc,
        )
        # result 保持默认值的 calendar_query_ok=False

    return result


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
    """Aggregate daily bars into completed Friday-anchored weekly bars.

    Each output row includes:
      - trading_days_count: number of daily bars in that week
      - last_daily_date:     latest daily date that contributed to the bar
    """
    daily = ensure_daily_frame(df)
    if cutoff_date is not None:
        daily = daily[daily["date"] <= cutoff_date]

    if daily.empty:
        return pd.DataFrame(
            columns=[
                "date", "open", "high", "low", "close", "volume",
                "trading_days_count", "last_daily_date",
            ]
        )

    daily_indexed = daily.set_index("date")
    resampler = daily_indexed.resample("W-FRI")

    ohlcv_agg = resampler.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trading_days_count=("open", "count"),
    )

    last_daily = (
        daily_indexed.index.to_series()
        .resample("W-FRI")
        .max()
        .rename("last_daily_date")
    )

    weekly = (
        pd.concat([ohlcv_agg, last_daily], axis=1)
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
        return pd.DataFrame(
            columns=[
                "date", "open", "high", "low", "close", "volume",
                "trading_days_count", "last_daily_date",
            ]
        )

    daily_latest_date = pd.to_datetime(daily["date"]).max()

    # Determine cutoff date based on current time and available daily data
    cutoff_date = get_last_completed_week_end(now, daily_latest_date)
    return convert_daily_to_weekly(df, cutoff_date=cutoff_date)
