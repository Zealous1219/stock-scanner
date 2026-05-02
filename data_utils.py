"""周线转换、交易日历查询、周线完成判定工具函数。

核心路径：
  scanner  : get_completed_weekly_bars(df, now=context.now)
             → get_last_completed_week_end(now, daily_latest_date)
             → convert_daily_to_weekly(df, cutoff_date)
  replay   : precompute_weekly_bars_for_replay(full_df)
             → slice_weekly_bars_for_snapshot(weekly_full, snapshot_date, ...)

作者: zealous
"""

from __future__ import annotations

import logging
from datetime import datetime

import baostock as bs
import pandas as pd

logger = logging.getLogger(__name__)


def get_week_monday_friday(target_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """返回指定日期所在自然周的周一和周五（W-FRI 锚点）。"""
    this_monday = target_date - pd.Timedelta(days=target_date.weekday())
    this_friday = this_monday + pd.Timedelta(days=4)
    return this_monday, this_friday


def get_snapshot_trading_week_info(snapshot_date: datetime | pd.Timestamp) -> dict:
    """返回 snapshot 对应自然周的交易日历元数据。

    用于区分"真实交易完成的周"与"日历上凑出来的周锚点"。

    Returns dict with keys:
      snapshot_date, friday_anchor, week_start_date, week_end_date,
      last_trading_day_of_week, has_trading_day, is_valid_completed_trading_week,
      calendar_query_ok, trading_days_count
    """
    snap_ts = pd.Timestamp(snapshot_date).normalize()
    week_start, week_end = get_week_monday_friday(snap_ts)

    result = {
        "snapshot_date": snap_ts,
        "friday_anchor": snap_ts,
        "week_start_date": week_start,
        "week_end_date": week_end,
        "last_trading_day_of_week": None,
        "has_trading_day": False,
        "is_valid_completed_trading_week": False,
        "calendar_query_ok": False,
        "trading_days_count": 0,
    }

    start_date = week_start.strftime("%Y-%m-%d")
    end_date = week_end.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            logger.error(
                "snapshot 日历查询失败 - snapshot_date=%s, range=[%s, %s], error_code=%s",
                snap_ts.date(), start_date, end_date, rs.error_code,
            )
            return result

        trading_days = []
        while rs.next():
            cal_date, is_trading = rs.get_row_data()
            if is_trading == "1":
                trading_days.append(pd.Timestamp(cal_date))

        result["calendar_query_ok"] = True
        result["trading_days_count"] = len(trading_days)
        if trading_days:
            result["has_trading_day"] = True
            result["last_trading_day_of_week"] = max(trading_days)
            result["is_valid_completed_trading_week"] = True

    except Exception as exc:
        logger.error(
            "snapshot 日历查询异常 - snapshot_date=%s, range=[%s, %s], exc=%s",
            snap_ts.date(), start_date, end_date, exc,
        )

    return result


def get_last_trading_day_of_week(target_date: pd.Timestamp) -> pd.Timestamp | None:
    """返回本周最后一个交易日。查询失败或整周无交易则返回 None。"""
    mon, fri = get_week_monday_friday(target_date)
    start = mon.strftime("%Y-%m-%d")
    end = fri.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        if rs.error_code != "0":
            logger.error("日历查询失败 - target_date=%s, error_code=%s", target_date.date(), rs.error_code)
            return None
        trading_days = []
        while rs.next():
            d, is_trading = rs.get_row_data()
            if is_trading == "1":
                trading_days.append(pd.Timestamp(d))
        if trading_days:
            return max(trading_days)
        logger.warning("本周无交易日 - %s", target_date.date())
        return None
    except Exception as exc:
        logger.error("日历查询异常 - target_date=%s, exc=%s", target_date.date(), exc)
        return None


def ensure_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """标准化日线 DataFrame：date 转 datetime，数值列转 numeric，去空。"""
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized.dropna(subset=["date", "open", "high", "low", "close", "volume"])


def get_last_completed_week_end(
    now: datetime | None = None,
    daily_latest_date: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """返回最新的已完成交易周的周五锚点。

    规则：
      - 周一~周四 → 上周五
      - 周五 < 20:00 → 上周五
      - 周五 ≥ 20:00 → 若 daily_latest_date ≥ 本周最后交易日则本周五，否则上周五
      - 周六/日 → 同周五规则
    """
    if now is None:
        now = datetime.now()

    current_date = pd.Timestamp(now.date())
    weekday = current_date.weekday()
    hour = now.hour

    _, this_friday = get_week_monday_friday(current_date)
    prev_friday = this_friday - pd.Timedelta(days=7)

    if weekday < 4:
        logger.debug("周一~周四，回退上周五: %s", prev_friday.date())
        return prev_friday

    if weekday == 4:
        if hour < 20:
            logger.debug("周五 20:00 前，回退上周五: %s", prev_friday.date())
            return prev_friday
        if daily_latest_date is not None:
            last_trading = get_last_trading_day_of_week(current_date)
            if last_trading is None:
                logger.warning("日历查询失败，回退上周五: %s", prev_friday.date())
                return prev_friday
            if daily_latest_date >= last_trading:
                logger.debug("本周完成，返回本周五: %s", this_friday.date())
                return this_friday
            logger.debug("数据未到本周最后交易日，回退上周五: %s", prev_friday.date())
            return prev_friday
        logger.warning("daily_latest_date 为 None，回退上周五: %s", prev_friday.date())
        return prev_friday

    # 周六/日
    if daily_latest_date is not None:
        last_trading = get_last_trading_day_of_week(current_date)
        if last_trading is None:
            logger.warning("日历查询失败，回退上周五: %s", prev_friday.date())
            return prev_friday
        if daily_latest_date >= last_trading:
            return this_friday
        return prev_friday
    return prev_friday


def convert_daily_to_weekly(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """日线聚合为 W-FRI 锚点的周线。

    每根周线包含：
      - trading_days_count: 该周实际交易天数
      - last_daily_date:    该周最后交易日
    """
    daily = ensure_daily_frame(df)
    if cutoff_date is not None:
        daily = daily[daily["date"] <= cutoff_date]
    if daily.empty:
        return pd.DataFrame(columns=[
            "date", "open", "high", "low", "close", "volume",
            "trading_days_count", "last_daily_date",
        ])

    daily_idx = daily.set_index("date")
    resampler = daily_idx.resample("W-FRI")

    ohlcv = resampler.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trading_days_count=("open", "count"),
    )
    last_daily = (
        daily_idx.index.to_series()
        .resample("W-FRI")
        .max()
        .rename("last_daily_date")
    )
    weekly = pd.concat([ohlcv, last_daily], axis=1).dropna().reset_index()
    return weekly


def get_completed_weekly_bars(
    df: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    """返回已完成周线（忽略当前未完成周）。

    通过 get_last_completed_week_end 确定截止日期。
    """
    daily = ensure_daily_frame(df)
    if daily.empty:
        return pd.DataFrame(columns=[
            "date", "open", "high", "low", "close", "volume",
            "trading_days_count", "last_daily_date",
        ])
    daily_latest = pd.to_datetime(daily["date"]).max()
    cutoff = get_last_completed_week_end(now, daily_latest)
    return convert_daily_to_weekly(df, cutoff_date=cutoff)
