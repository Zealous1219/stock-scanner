"""Config-driven stock scanner application."""

from __future__ import annotations

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import baostock as bs
import pandas as pd

from config_loader import load_config
from data_utils import ensure_daily_frame, get_last_trading_day_of_week
from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult
from strategies import create_strategy_from_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
OUTPUT_DIR = "output"
VALIDATION_DIR = "validation"


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def should_force_refresh_on_friday(now: datetime) -> bool:
    """Check if we should force refresh data on Friday after 20:00.

    Returns True if it's Friday after 20:00, False otherwise.
    """
    return now.weekday() == 4 and now.hour >= 20  # Friday (weekday 4) after 20:00


def get_latest_trading_day(reference_time: datetime | None = None, lookback_days: int = 14) -> str:
    """Return the latest trading day on or before the reference date."""
    if reference_time is None:
        reference_time = datetime.now()

    end_date = reference_time.strftime("%Y-%m-%d")
    start_date = (reference_time - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)

    if rs.error_code != "0":
        raise RuntimeError(f"Failed to query trade dates: {rs.error_msg}")

    latest_trading_day = None
    while rs.next():
        calendar_date, is_trading_day = rs.get_row_data()
        if is_trading_day == "1":
            latest_trading_day = calendar_date

    if latest_trading_day is None:
        raise RuntimeError(f"No trading day found between {start_date} and {end_date}.")

    return latest_trading_day


def get_stock_list(stock_pool: str) -> List[str]:
    pool_mapping = {
        "hs300": ("query_hs300_stocks", "沪深300"),
        "zz500": ("query_zz500_stocks", "中证500"),
        "sz50": ("query_sz50_stocks", "上证50"),
        "all": ("query_all_stock", "全部A股"),
    }

    pool_key = stock_pool.lower()
    if pool_key not in pool_mapping:
        raise ValueError(f"Unsupported stock pool: {stock_pool}")

    api_name, pool_name = pool_mapping[pool_key]
    logger.info("Fetching stock list for %s...", pool_name)

    if pool_key == "all":
        query_day = get_latest_trading_day()
        logger.info("Using trading day %s for all-stock pool", query_day)
        rs = getattr(bs, api_name)(day=query_day)
    else:
        rs = getattr(bs, api_name)()

    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError(f"Failed to fetch stock list for pool {stock_pool}")

    df = pd.DataFrame(data_list, columns=rs.fields)

    # 记录实际获取的股票数量
    logger.info("Fetched %s stocks for %s", len(df), pool_name)

    # 数量校验
    stock_count = len(df)
    if pool_key == "hs300":
        if stock_count < 250 or stock_count > 350:
            logger.error("baostock returned unexpected stock count for pool %s (hs300): got %s, expected 250-350", pool_name, stock_count)
            raise RuntimeError(f"baostock returned unexpected stock count for pool {pool_name} (hs300): got {stock_count}, expected 250-350")
    elif pool_key == "zz500":
        if stock_count < 450 or stock_count > 600:
            logger.error("baostock returned unexpected stock count for pool %s (zz500): got %s, expected 450-600", pool_name, stock_count)
            raise RuntimeError(f"baostock returned unexpected stock count for pool {pool_name} (zz500): got {stock_count}, expected 450-600")
    elif pool_key == "sz50":
        if stock_count < 45 or stock_count > 60:
            logger.error("baostock returned unexpected stock count for pool %s (sz50): got %s, expected 45-60", pool_name, stock_count)
            raise RuntimeError(f"baostock returned unexpected stock count for pool {pool_name} (sz50): got {stock_count}, expected 45-60")
    # all 不校验数量

    return df["code"].tolist()


def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    try:
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,code,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
        )

        data_list = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        return ensure_daily_frame(pd.DataFrame(data_list, columns=rs.fields))
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", symbol, exc)
        return None


def load_or_update_data(
    symbol: str,
    lookback_days: int,
    initial_days: int,
    request_interval: float,
) -> Tuple[pd.DataFrame | None, bool]:
    filename = symbol.replace(".", "_")
    file_path = os.path.join(DATA_DIR, f"{filename}.csv")

    today = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=initial_days)).strftime("%Y-%m-%d")
    performed_remote_fetch = False

    if not os.path.exists(file_path):
        df = fetch_historical_data(symbol, start_date, today)
        performed_remote_fetch = True
        if df is None:
            return None, performed_remote_fetch
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df.tail(lookback_days), performed_remote_fetch

    try:
        local_df = ensure_daily_frame(pd.read_csv(file_path))
        if len(local_df) < 60:
            df = fetch_historical_data(symbol, start_date, today)
            performed_remote_fetch = True
            if df is None:
                return None, performed_remote_fetch
            df.to_csv(file_path, index=False)
            if request_interval > 0:
                time.sleep(request_interval)
            return df.tail(lookback_days), performed_remote_fetch

        latest_date = pd.to_datetime(local_df["date"]).max()
        yesterday = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
        today_timestamp = pd.Timestamp(datetime.now().date())
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Check if we should force refresh on Friday after 20:00
        now = datetime.now()
        force_refresh = should_force_refresh_on_friday(now)

        if force_refresh:
            # Friday after 20:00: only refresh if we don't have today's data
            if latest_date >= today_timestamp:
                # Already have today's data, use cache
                return local_df.tail(lookback_days), performed_remote_fetch
            # Otherwise, proceed to fetch new data
        elif latest_date >= yesterday:
            # Not Friday after 20:00, use normal cache logic
            return local_df.tail(lookback_days), performed_remote_fetch

        new_data = fetch_historical_data(symbol, latest_date.strftime("%Y-%m-%d"), today_str)
        performed_remote_fetch = True
        if new_data is None or new_data.empty:
            return local_df.tail(lookback_days), performed_remote_fetch

        combined_df = (
            pd.concat([local_df, new_data], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        combined_df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return combined_df.tail(lookback_days), performed_remote_fetch
    except Exception as exc:
        logger.warning("%s cache read failed (%s), refetching", symbol, exc)
        df = fetch_historical_data(symbol, start_date, today)
        performed_remote_fetch = True
        if df is None:
            return None, performed_remote_fetch
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df.tail(lookback_days), performed_remote_fetch


def write_run_log(
    strategy_name: str,
    stock_pool: str,
    decision: StrategyDecision,
    config: Dict[str, Any],
    metrics: Dict[str, Any] | None = None,
) -> str:
    metrics = metrics or {}
    run_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(OUTPUT_DIR, f"strategy_runs_{run_date}.csv")

    row = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_name": strategy_name,
        "stock_pool": stock_pool,
        "execution_status": "executed" if decision.should_run else "skipped",
        "reason_code": decision.reason_code,
        "reason_text": decision.reason_text,
        "config_snapshot": str(config),
    }
    row.update(metrics)

    run_df = pd.DataFrame([row])
    if os.path.exists(output_file):
        run_df.to_csv(output_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        run_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return output_file


def build_candidate_row(symbol: str, result: StrategyResult, df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    row = {
        "code": symbol,
        "strategy": result.details.get("signal_type", "matched"),
        "latest_daily_date": str(pd.to_datetime(latest["date"]).date()),
        "latest_close": float(latest["close"]),
        "reason_code": result.reason_code,
        "reason_text": result.reason_text,
    }
    row.update(result.details)
    return row


def write_candidates(strategy_name: str, results: List[Dict[str, Any]]) -> str:
    run_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(OUTPUT_DIR, f"{strategy_name}_candidates_{run_date}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file

#### 主入口  
def main() -> None:
    ensure_directories()
    logger.info("=" * 60)
    logger.info("Stock scanner starting")
    logger.info("=" * 60)

    config = load_config()
    strategy_config = config.get("strategy", {})
    env_strategy_name = os.getenv("STRATEGY_NAME")
    if env_strategy_name:
        # 当 STRATEGY_NAME 被设置时，直接从目标策略类读取 DEFAULT_PARAMS 作为参数
        # 避免旧策略参数污染新策略，也避免通过实例化触发初始化和参数校验
        try:
            import importlib
            from strategies import BUILTIN_STRATEGIES

            normalized_name = env_strategy_name.lower()
            strategy_class = BUILTIN_STRATEGIES.get(normalized_name)

            if strategy_class is None:
                try:
                    module = importlib.import_module(f'strategies.{normalized_name}')
                    strategy_class = getattr(module, 'Strategy')
                except (ImportError, AttributeError):
                    pass

            if strategy_class and hasattr(strategy_class, 'DEFAULT_PARAMS'):
                default_params = strategy_class.DEFAULT_PARAMS.copy()
                strategy_config = {
                    **strategy_config,
                    "name": env_strategy_name,
                    "params": default_params
                }
                logger.info("Using strategy override from STRATEGY_NAME=%s", env_strategy_name)
                logger.info("Using target strategy's DEFAULT_PARAMS as params")
            else:
                strategy_config = {**strategy_config, "name": env_strategy_name}
                logger.info("Using strategy override from STRATEGY_NAME=%s (name only)", env_strategy_name)

        except Exception as e:
            logger.warning(f"Failed to load strategy class {env_strategy_name}: {e}")
            strategy_config = {**strategy_config, "name": env_strategy_name}
            logger.info("Using strategy override from STRATEGY_NAME=%s (fallback: name only)", env_strategy_name)
    env_stock_pool = os.getenv("STOCK_POOL")
    stock_pool = env_stock_pool or config.get("stock_pool", {}).get("type", "hs300")
    if env_stock_pool:
        logger.info("Using stock pool override from STOCK_POOL=%s", env_stock_pool)
    data_config = config.get("data", {})

    lookback_days = int(data_config.get("lookback_days", 180))
    initial_days = int(data_config.get("initial_days", 400))
    request_interval = float(data_config.get("request_interval", 0.5))

    strategy = create_strategy_from_config(strategy_config)
    context = StrategyContext(now=datetime.now(), stock_pool=stock_pool, config=config)
    decision = strategy.can_run(context)

    if not decision.should_run:
        log_file = write_run_log(strategy.name, stock_pool, decision, config)
        logger.info("Strategy skipped: %s", decision.reason_text)
        logger.info("Run log written to %s", log_file)
        return

    logger.info("Logging in to baostock...")
    lg = bs.login()
    logger.info("baostock login: %s", lg.error_msg)

    if lg.error_code != "0":
        logger.error("baostock login failed: error_code=%s, error_msg=%s", lg.error_code, lg.error_msg)
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    try:
        stocks = get_stock_list(stock_pool)
        logger.info("Processing %s symbols with strategy %s", len(stocks), strategy.name)

        results: List[Dict[str, Any]] = []
        success_count = 0
        fail_count = 0
        remote_fetch_count = 0

        for index, symbol in enumerate(stocks, start=1):
            if index == 1 or index % 100 == 0:
                logger.info(
                    "Progress %s/%s | success=%s fail=%s matched=%s",
                    index,
                    len(stocks),
                    success_count,
                    fail_count,
                    len(results),
                )

            df, fetched = load_or_update_data(symbol, lookback_days, initial_days, request_interval)
            if fetched:
                remote_fetch_count += 1

            if df is None or df.empty:
                fail_count += 1
                continue

            success_count += 1
            result = strategy.scan(symbol, df, context)
            if result.matched:
                results.append(build_candidate_row(symbol, result, df))

        candidate_file = ""
        if results:
            candidate_file = write_candidates(strategy_config.get("name", strategy.name), results)
            logger.info("Candidates written to %s", candidate_file)
        else:
            logger.info("No symbols matched strategy %s", strategy.name)

        log_file = write_run_log(
            strategy.name,
            stock_pool,
            decision,
            config,
            metrics={
                "processed_symbols": len(stocks),
                "successful_symbols": success_count,
                "failed_symbols": fail_count,
                "matched_symbols": len(results),
                "remote_fetch_count": remote_fetch_count,
                "candidate_file": candidate_file,
            },
        )
        logger.info("Run log written to %s", log_file)
    finally:
        bs.logout()
        logger.info("Scanner finished")


def load_full_df_for_replay(
    symbol: str,
    lookback_days: int,
    initial_days: int,
    request_interval: float,
    required_history_days: int,
    replay_data_end_date: str | None = None,
) -> pd.DataFrame | None:
    """Load complete, standardized full_df for a stock for replay.

    Args:
        replay_data_end_date: Fixed end date for replay data (YYYY-MM-DD string).
            When provided, this replaces datetime.now() as the data boundary,
            ensuring replay reproducibility across runs.

    Returns DataFrame with datetime date column, or None if load failed.
    """
    filename = symbol.replace(".", "_")
    file_path = os.path.join(DATA_DIR, f"{filename}.csv")

    history_days = max(initial_days, required_history_days)

    if replay_data_end_date is not None:
        end_date_obj = datetime.strptime(replay_data_end_date, "%Y-%m-%d").date()
        end_date = replay_data_end_date
        start_date_obj = end_date_obj - timedelta(days=history_days)
        start_date = start_date_obj.strftime("%Y-%m-%d")
    else:
        end_date_obj = datetime.now().date()
        end_date = end_date_obj.strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")

    if not os.path.exists(file_path):
        df = fetch_historical_data(symbol, start_date, end_date)
        if df is None:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df

    try:
        local_df = ensure_daily_frame(pd.read_csv(file_path))
        local_df["date"] = pd.to_datetime(local_df["date"])

        required_start = pd.Timestamp(end_date_obj - timedelta(days=history_days))

        earliest_date = local_df["date"].min()
        if earliest_date > required_start:
            earliest_date_str = earliest_date.strftime("%Y-%m-%d")
            fetch_end = (earliest_date - timedelta(days=1)).strftime("%Y-%m-%d")
            old_data = fetch_historical_data(symbol, start_date, fetch_end)
            if old_data is not None and not old_data.empty:
                old_data["date"] = pd.to_datetime(old_data["date"])
                local_df = (
                    pd.concat([old_data, local_df], ignore_index=True)
                    .drop_duplicates(subset=["date"])
                    .sort_values("date")
                )
                local_df.to_csv(file_path, index=False)
                if request_interval > 0:
                    time.sleep(request_interval)

        latest_date = local_df["date"].max()
        today_timestamp = pd.Timestamp(end_date_obj)
        today_str = end_date

        if latest_date >= today_timestamp:
            return local_df

        new_data = fetch_historical_data(symbol, latest_date.strftime("%Y-%m-%d"), today_str)
        if new_data is None or new_data.empty:
            return local_df

        new_data["date"] = pd.to_datetime(new_data["date"])
        combined_df = (
            pd.concat([local_df, new_data], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        combined_df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return combined_df

    except Exception as exc:
        logger.warning("%s full_df load failed (%s), refetching", symbol, exc)
        df = fetch_historical_data(symbol, start_date, end_date)
        if df is None:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df


def generate_weekly_snapshot_dates(lookback_weeks: int) -> List[datetime]:
    """Generate exactly ``lookback_weeks`` completed Friday snapshot dates.

    Each snapshot represents the moment after a *completed* weekly trading
    week.  The current week is excluded when its last trading day has not
    yet passed (or on that day before 20:00).  Every included Friday is
    anchored at 23:59:59 so that the weekly cutoff logic in
    get_last_completed_week_end sees hour >= 20 consistently, regardless
    of when the replay is actually run.

    The date list is built by decrementing from the last completed Friday:
    this guarantees that ``lookback_weeks=N`` produces exactly N snapshots
    rather than N or N+1 depending on whether the window interval happens
    to land on a Friday.

    Args:
        lookback_weeks: Number of completed weekly snapshots to generate.
                        ``lookback_weeks=52`` returns exactly 52 snapshots.
    """
    now = datetime.now()
    today = now.date()
    weekday = now.weekday()

    # 1. Determine this week's Friday anchor (natural week)
    if weekday < 4:  # Monday - Thursday
        this_friday_anchor = today + timedelta(days=4 - weekday)
    elif weekday == 4:  # Friday
        this_friday_anchor = today
    else:  # Saturday (5) or Sunday (6)
        this_friday_anchor = today - timedelta(days=weekday - 4)

    # 2. Query the last trading day of this week using the trading calendar
    last_trading_day = None
    try:
        last_trading_day = get_last_trading_day_of_week(pd.Timestamp(today))
    except Exception:
        # Any unexpected exception from the calendar query is treated as
        # a failure — we fall back conservatively below.
        pass

    # 3. Decide whether this week is completed
    if last_trading_day is None:
        # Calendar query failed — be conservative and fall back to the
        # previous Friday anchor.
        last_completed_friday = this_friday_anchor - timedelta(weeks=1)
    else:
        last_trading_date = last_trading_day.date()
        if today < last_trading_date:
            # Current date is before the last trading day of this week
            last_completed_friday = this_friday_anchor - timedelta(weeks=1)
        elif today > last_trading_date:
            # Current date is after the last trading day of this week
            last_completed_friday = this_friday_anchor
        else:
            # today == last_trading_date
            if now.hour < 20:
                last_completed_friday = this_friday_anchor - timedelta(weeks=1)
            else:
                last_completed_friday = this_friday_anchor

    # Build exactly lookback_weeks snapshots by decrementing from the last
    # completed Friday.  Decrementing by whole weeks guarantees we always
    # land on a Friday.
    friday_dates = []
    for i in range(lookback_weeks):
        friday = last_completed_friday - timedelta(weeks=i)
        snapshot = datetime.combine(
            friday,
            datetime.min.time().replace(hour=23, minute=59, second=59),
        )
        friday_dates.append(snapshot)

    friday_dates.sort()
    return friday_dates


def load_historical_data_up_to_date(
    symbol: str,
    cutoff_date: datetime,
    lookback_days: int,
    initial_days: int,
    request_interval: float,
    full_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Load historical data up to (and including) the cutoff date.

    If full_df is provided, uses memory filtering instead of reading from disk.
    """
    if full_df is not None:
        filtered_df = full_df[full_df["date"] <= pd.Timestamp(cutoff_date)].copy()
        if not filtered_df.empty:
            return filtered_df.tail(lookback_days)
        return None

    filename = symbol.replace(".", "_")
    file_path = os.path.join(DATA_DIR, f"{filename}.csv")

    start_date = (cutoff_date - timedelta(days=initial_days)).strftime("%Y-%m-%d")
    end_date = cutoff_date.strftime("%Y-%m-%d")

    if not os.path.exists(file_path):
        df = fetch_historical_data(symbol, start_date, end_date)
        if df is None:
            return None
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df

    try:
        local_df = ensure_daily_frame(pd.read_csv(file_path))

        if len(local_df) < 60:
            df = fetch_historical_data(symbol, start_date, end_date)
            if df is None:
                return None
            df.to_csv(file_path, index=False)
            if request_interval > 0:
                time.sleep(request_interval)
            return df

        local_df["date"] = pd.to_datetime(local_df["date"])
        filtered_df = local_df[local_df["date"] <= pd.Timestamp(cutoff_date)]

        latest_date = filtered_df["date"].max() if not filtered_df.empty else None

        if latest_date is None or latest_date < pd.Timestamp(cutoff_date.date()):
            fetch_start = latest_date.strftime("%Y-%m-%d") if latest_date else start_date
            new_data = fetch_historical_data(symbol, fetch_start, end_date)
            if new_data is not None and not new_data.empty:
                combined_df = (
                    pd.concat([local_df, new_data], ignore_index=True)
                    .drop_duplicates(subset=["date"])
                    .sort_values("date")
                )
                combined_df.to_csv(file_path, index=False)
                if request_interval > 0:
                    time.sleep(request_interval)
                filtered_df = combined_df[combined_df["date"] <= pd.Timestamp(cutoff_date)]

        return filtered_df.tail(lookback_days)

    except Exception as exc:
        logger.warning("%s历史数据读取失败 (%s)，重新获取", symbol, exc)
        df = fetch_historical_data(symbol, start_date, end_date)
        if df is None:
            return None
        df.to_csv(file_path, index=False)
        if request_interval > 0:
            time.sleep(request_interval)
        return df


def calculate_forward_returns(
    symbol: str,
    result: StrategyResult,
    full_df: pd.DataFrame | None = None,
) -> Dict[str, float]:
    """Calculate 4-week, 8-week, and 12-week forward returns.

    If full_df is provided, uses it instead of reading from disk.
    """
    forward_returns = {
        "return_4w": float("nan"),
        "return_8w": float("nan"),
        "return_12w": float("nan"),
        "return_16w": float("nan"),
        "return_20w": float("nan"),
    }
    max_gap_days = 3

    signal_date_str = result.details.get("signal_date", "")
    if not signal_date_str:
        return forward_returns

    try:
        signal_date = pd.to_datetime(signal_date_str).date()

        future_dates = {
            "4w": signal_date + timedelta(weeks=4),
            "8w": signal_date + timedelta(weeks=8),
            "12w": signal_date + timedelta(weeks=12),
            "16w": signal_date + timedelta(weeks=16),
            "20w": signal_date + timedelta(weeks=20),
        }

        if full_df is not None:
            df = full_df.copy()
        else:
            filename = symbol.replace(".", "_")
            file_path = os.path.join(DATA_DIR, f"{filename}.csv")

            if not os.path.exists(file_path):
                return forward_returns

            df = ensure_daily_frame(pd.read_csv(file_path))
            if df.empty:
                return forward_returns

        df["date"] = pd.to_datetime(df["date"])

        signal_data = df[df["date"] == pd.Timestamp(signal_date)]
        if signal_data.empty:
            return forward_returns

        signal_close = float(signal_data.iloc[0]["close"])

        for period, future_date in future_dates.items():
            target_ts = pd.Timestamp(future_date)
            exact_match = df[df["date"] == target_ts]

            if not exact_match.empty:
                future_row = exact_match.iloc[0]
            elif df.empty:
                continue
            else:
                date_deltas = (df["date"] - target_ts).abs()
                nearest_idx = date_deltas.idxmin()
                nearest_gap = date_deltas.loc[nearest_idx]

                if pd.isna(nearest_gap) or nearest_gap > pd.Timedelta(days=max_gap_days):
                    logger.warning(
                        "%s %s forward return target %s has no trading day within %d days; keeping NaN",
                        symbol,
                        period,
                        target_ts.strftime("%Y-%m-%d"),
                        max_gap_days,
                    )
                    continue

                future_row = df.loc[nearest_idx]

            future_close = float(future_row["close"])

            returns = (future_close - signal_close) / signal_close

            if period == "4w":
                forward_returns["return_4w"] = returns
            elif period == "8w":
                forward_returns["return_8w"] = returns
            elif period == "12w":
                forward_returns["return_12w"] = returns
            elif period == "16w":
                forward_returns["return_16w"] = returns
            elif period == "20w":
                forward_returns["return_20w"] = returns

    except Exception as exc:
        logger.debug("计算%s前向收益率失败: %s", symbol, exc)
        raise  # Re-raise to be caught by per-symbol exception handler

    return forward_returns


def build_replay_record(
    symbol: str,
    result: StrategyResult,
    context: StrategyContext,
    forward_returns: Dict[str, float],
    experiment_tag: str,
    universe: str,
    lookback_weeks: int,
) -> Dict[str, Any]:
    """Build a replay record with core signal fields and forward returns."""
    details = result.details

    def extract_date_string(date_value):
        if not date_value:
            return ""
        if isinstance(date_value, str):
            if " " in date_value:
                return date_value.split(" ")[0]
            return date_value
        try:
            if isinstance(date_value, pd.Timestamp):
                return date_value.strftime("%Y-%m-%d")
        except:
            pass
        str_value = str(date_value)
        if " " in str_value:
            return str_value.split(" ")[0]
        return str_value

    record = {
        "experiment_tag": experiment_tag,
        "universe": universe,
        "lookback_weeks": lookback_weeks,
        "code": symbol,
        "strategy": details.get("signal_type", "matched"),
        "snapshot_date": context.now.strftime("%Y-%m-%d"),
        "signal_date": extract_date_string(details.get("signal_date", "")),
        "reason_code": result.reason_code,
        "signal_type": details.get("signal_type", ""),
        "downtrend_weeks": details.get("downtrend_weeks", ""),
        "big1_date": extract_date_string(details.get("big1_date", details.get("big1", {}).get("date", ""))),
        "small1_date": extract_date_string(details.get("small1_date", details.get("small1", {}).get("date", ""))),
        "small2_date": extract_date_string(details.get("small2_date", details.get("small2", {}).get("date", ""))),
        "small3_date": extract_date_string(details.get("small3_date", details.get("small3", {}).get("date", ""))),
        "pivot_bar_date": extract_date_string(details.get("pivot_bar_date", details.get("pivot_bar", {}).get("date", ""))),
        "forward_4w_return": forward_returns.get("return_4w", float("nan")),
        "forward_8w_return": forward_returns.get("return_8w", float("nan")),
        "forward_12w_return": forward_returns.get("return_12w", float("nan")),
        "forward_16w_return": forward_returns.get("return_16w", float("nan")),
        "forward_20w_return": forward_returns.get("return_20w", float("nan")),
    }

    return record


# Replay result CSV columns (must match build_replay_record output)
_REPLAY_RESULT_COLUMNS = [
    "experiment_tag",
    "universe",
    "lookback_weeks",
    "code",
    "strategy",
    "snapshot_date",
    "signal_date",
    "reason_code",
    "signal_type",
    "downtrend_weeks",
    "big1_date",
    "small1_date",
    "small2_date",
    "small3_date",
    "pivot_bar_date",
    "forward_4w_return",
    "forward_8w_return",
    "forward_12w_return",
    "forward_16w_return",
    "forward_20w_return",
]


def write_replay_results(
    results: List[Dict[str, Any]],
    universe: str,
    lookback_weeks: int,
    version: str,
    snapshot_date: str,
) -> str:
    """Write replay validation results to a per-snapshot CSV file.

    Each snapshot gets its own result file. Re-running the same snapshot
    overwrites the file instead of appending, preventing duplicate rows.
    Even empty results will produce a CSV with headers for schema stability.

    Args:
        results: List of replay records to write
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        snapshot_date: Snapshot date in YYYY-MM-DD format
    """
    replay_file = get_replay_snapshot_result_path(universe, lookback_weeks, version, snapshot_date)

    if results:
        df = pd.DataFrame(results)
    else:
        # Write empty CSV with headers for schema stability
        df = pd.DataFrame(columns=_REPLAY_RESULT_COLUMNS)
    df.to_csv(replay_file, index=False, encoding="utf-8-sig")
    logger.info("Replay results written to %s (%d rows)", replay_file, len(df))
    return replay_file


def write_replay_errors(
    errors: List[Dict[str, Any]],
    universe: str,
    lookback_weeks: int,
    version: str,
    snapshot_date: str,
) -> str:
    """Write replay error records to a per-snapshot CSV file.

    Each snapshot gets its own error file. Re-running the same snapshot
    overwrites the file instead of appending, preventing duplicate rows.
    If errors is empty, the old error file (if any) is removed to prevent
    stale error records from persisting.

    Args:
        errors: List of error records to write
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        snapshot_date: Snapshot date in YYYY-MM-DD format
    """
    error_file = get_replay_snapshot_error_path(universe, lookback_weeks, version, snapshot_date)

    if not errors:
        # No errors this run — remove stale error file if it exists
        try:
            if os.path.exists(error_file):
                os.remove(error_file)
                logger.info("Stale error file removed: %s", error_file)
        except OSError as exc:
            logger.warning("Failed to remove stale error file %s: %s", error_file, exc)
        return error_file

    df = pd.DataFrame(errors)
    df.to_csv(error_file, index=False, encoding="utf-8-sig")
    logger.info("Replay errors written to %s (%d records)", error_file, len(df))

    return error_file


def get_replay_snapshot_result_path(universe: str, lookback_weeks: int, version: str, snapshot_date: str) -> str:
    """Return the per-snapshot result file path for a replay experiment.

    Args:
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        snapshot_date: Snapshot date in YYYY-MM-DD format
    """
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{universe}_{lookback_weeks}w_{version}_{snapshot_date}.csv",
    )


def get_replay_snapshot_error_path(universe: str, lookback_weeks: int, version: str, snapshot_date: str) -> str:
    """Return the per-snapshot error file path for a replay experiment.

    Args:
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        snapshot_date: Snapshot date in YYYY-MM-DD format
    """
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{universe}_{lookback_weeks}w_{version}_{snapshot_date}_errors.csv",
    )


def get_replay_checkpoint_path(universe: str, lookback_weeks: int, version: str) -> str:
    """Return the checkpoint path for a replay experiment."""
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{universe}_{lookback_weeks}w_{version}.checkpoint.json",
    )


def _verify_completed_snapshot_files(
    completed_snapshots: List[str],
    universe: str,
    lookback_weeks: int,
    version: str,
) -> None:
    """Verify that all completed snapshots have their result files.

    This is a conservative check: if the checkpoint claims a snapshot is
    completed but the corresponding result file is missing, we refuse to
    resume to avoid inconsistent state.

    Args:
        completed_snapshots: List of completed snapshot dates (YYYY-MM-DD)
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier

    Raises:
        RuntimeError: If any completed snapshot is missing its result file.
    """
    missing_files = []

    for snapshot_date in completed_snapshots:
        result_path = get_replay_snapshot_result_path(
            universe, lookback_weeks, version, snapshot_date
        )
        if not os.path.exists(result_path):
            missing_files.append((snapshot_date, result_path))

    if missing_files:
        missing_str = ", ".join(
            f"{date} ({path})" for date, path in missing_files
        )
        raise RuntimeError(
            "Checkpoint marks the following snapshots as completed, "
            "but their result files are missing. "
            "Refusing to resume due to inconsistent state. "
            f"Missing files: {missing_str}. "
            "Please delete the checkpoint and run a fresh replay."
        )


def _summarize_failed_symbols(failed_symbols_per_snapshot: Dict[str, List[str]]) -> str:
    """Build a human-readable summary of unresolved failed symbols.

    Returns a log-ready string or empty string when there are no unresolved
    failures.  Shows the total snapshot/symbol counts and up to 5 example
    snapshots with the first few failed symbols each.

    Args:
        failed_symbols_per_snapshot: dict of snapshot_date → failed symbols.
    """
    if not failed_symbols_per_snapshot:
        return ""
    total_snapshots = len(failed_symbols_per_snapshot)
    total_symbols = sum(len(v) for v in failed_symbols_per_snapshot.values())
    lines = [
        "[Replay] Unresolved failed symbols: "
        f"{total_snapshots} snapshot(s) with {total_symbols} total failed symbol(s)",
    ]
    for i, (snap, syms) in enumerate(sorted(failed_symbols_per_snapshot.items())):
        if i >= 5:
            remaining = total_snapshots - 5
            if remaining > 0:
                lines.append(f"  ... and {remaining} more snapshot(s)")
            break
        syms_preview = ", ".join(syms[:3])
        if len(syms) > 3:
            syms_preview += f", ... ({len(syms)} total)"
        lines.append(f"  [{snap}] {syms_preview}")
    return "\n".join(lines)


def load_replay_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a replay checkpoint from JSON."""
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_replay_checkpoint(
    checkpoint: Dict[str, Any],
    experiment_tag: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    expected_snapshot_dates: List[str],
    expected_replay_data_end_date: str | None = None,
) -> List[str]:
    """Validate checkpoint identity and return completed snapshots.

    Checkpoint is the only source of truth for completed snapshot state.
    CSV outputs are result carriers only and must not be used to infer progress.

    The snapshot window identity (``snapshot_dates``) is now validated so
    that a checkpoint created during an earlier run with a *different*
    snapshot window is never mistaken for the same experiment.  This
    prevents silently merging results from two different time windows.

    The ``replay_data_end_date`` is validated to ensure the same data
    boundary is used across resume runs, preventing data drift when
    the replay is resumed on a different day.
    """
    expected_fields = {
        "experiment_tag": experiment_tag,
        "universe": universe,
        "lookback_weeks": lookback_weeks,
        "version": version,
    }
    mismatches = []
    for key, expected_value in expected_fields.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}={actual_value!r} (expected {expected_value!r})")

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            "Replay checkpoint does not match current experiment parameters: "
            f"{mismatch_text}"
        )

    # Validate snapshot window identity: the full set of snapshot dates
    # must be identical between the checkpoint and the current run.
    # Without this, a checkpoint from a run weeks ago could be mistaken
    # as resume-compatible even though the generated snapshot windows
    # are different.
    checkpoint_snapshot_dates = checkpoint.get("snapshot_dates")
    if checkpoint_snapshot_dates is None:
        raise RuntimeError(
            "Replay checkpoint is missing 'snapshot_dates'. "
            "The checkpoint schema is insufficient to verify that the "
            "snapshot window is identical to the current run. "
            "Cannot safely resume — please delete the checkpoint or "
            "run a fresh replay."
        )
    if not isinstance(checkpoint_snapshot_dates, list) or \
       any(not isinstance(item, str) for item in checkpoint_snapshot_dates):
        raise RuntimeError(
            "Replay checkpoint has invalid 'snapshot_dates'; "
            "expected a list of YYYY-MM-DD strings."
        )
    if checkpoint_snapshot_dates != expected_snapshot_dates:
        c_first = checkpoint_snapshot_dates[0]
        c_last = checkpoint_snapshot_dates[-1]
        c_count = len(checkpoint_snapshot_dates)
        e_first = expected_snapshot_dates[0]
        e_last = expected_snapshot_dates[-1]
        e_count = len(expected_snapshot_dates)
        raise RuntimeError(
            "Replay checkpoint snapshot window does not match the current run. "
            f"Checkpoint window: {c_first} → {c_last} ({c_count} snapshots). "
            f"Current window:   {e_first} → {e_last} ({e_count} snapshots). "
            "Cannot safely resume — these are different snapshot windows."
        )

    # Validate replay_data_end_date: the data boundary must be identical
    # between the checkpoint and the current run to ensure reproducibility.
    checkpoint_replay_data_end_date = checkpoint.get("replay_data_end_date")
    if checkpoint_replay_data_end_date is None:
        raise RuntimeError(
            "Replay checkpoint is missing 'replay_data_end_date'. "
            "The checkpoint schema is insufficient to verify that the "
            "data boundary is identical to the current run. "
            "This is required for replay reproducibility. "
            "Cannot safely resume — please delete the checkpoint or "
            "run a fresh replay with the new schema."
        )
    if expected_replay_data_end_date is not None:
        if checkpoint_replay_data_end_date != expected_replay_data_end_date:
            raise RuntimeError(
                "Replay checkpoint 'replay_data_end_date' does not match the current run. "
                f"Checkpoint: {checkpoint_replay_data_end_date}. "
                f"Current run: {expected_replay_data_end_date}. "
                "Cannot safely resume — the data boundary has changed."
            )

    completed_snapshots = checkpoint.get("completed_snapshots")
    if not isinstance(completed_snapshots, list) or any(not isinstance(item, str) for item in completed_snapshots):
        raise RuntimeError("Replay checkpoint has invalid completed_snapshots; expected a list of YYYY-MM-DD strings")

    # Validate failed_symbols_per_snapshot if present.
    # This is a quality/omission truth (NOT a progress blocker).
    # A completed snapshot MAY have failed symbols — this is a valid state.
    failed_symbols = checkpoint.get("failed_symbols_per_snapshot")
    if failed_symbols is not None:
        if not isinstance(failed_symbols, dict):
            raise RuntimeError(
                "Replay checkpoint has invalid 'failed_symbols_per_snapshot'; "
                "expected a dict[str, list[str]]"
            )
        for key, value in failed_symbols.items():
            if not isinstance(key, str):
                raise RuntimeError(
                    "Replay checkpoint has invalid 'failed_symbols_per_snapshot'; "
                    "all keys must be strings (YYYY-MM-DD snapshot dates)"
                )
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise RuntimeError(
                    "Replay checkpoint has invalid 'failed_symbols_per_snapshot'; "
                    "all values must be lists of strings (symbol codes)"
                )

    return completed_snapshots


def save_replay_checkpoint(
    checkpoint_path: str,
    experiment_tag: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    completed_snapshots: List[str],
    snapshot_dates: List[str],
    replay_data_end_date: str | None = None,
    failed_symbols_per_snapshot: Dict[str, List[str]] | None = None,
) -> None:
    """Persist replay checkpoint JSON for snapshot-level resume.

    ``snapshot_dates`` encodes the full snapshot window identity so that
    a future resume can verify that it is resuming the *same* window,
    not a different window generated weeks later.

    ``replay_data_end_date`` fixes the data boundary for replay so that
    the same experiment always uses identical data, regardless of when
    the replay is resumed.

    ``failed_symbols_per_snapshot`` is the structured record of which
    symbols failed in each completed snapshot.  It is a quality/omission
    truth, NOT a progress blocker.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checkpoint: Dict[str, Any] = {
        "experiment_tag": experiment_tag,
        "universe": universe,
        "lookback_weeks": lookback_weeks,
        "version": version,
        "snapshot_dates": snapshot_dates,
        "completed_snapshots": completed_snapshots,
        "updated_at": now_str,
    }

    if replay_data_end_date is not None:
        checkpoint["replay_data_end_date"] = replay_data_end_date

    if failed_symbols_per_snapshot is not None and failed_symbols_per_snapshot:
        checkpoint["failed_symbols_per_snapshot"] = failed_symbols_per_snapshot

    if os.path.exists(checkpoint_path):
        existing_checkpoint = load_replay_checkpoint(checkpoint_path)
        checkpoint["created_at"] = existing_checkpoint.get("created_at", now_str)
        if "replay_data_end_date" not in checkpoint and "replay_data_end_date" in existing_checkpoint:
            checkpoint["replay_data_end_date"] = existing_checkpoint["replay_data_end_date"]
    else:
        checkpoint["created_at"] = now_str

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=True, indent=2)


# ---------------------------------------------------------------------------
# Weekly-strategy replay optimization v1
#
# These helpers exist exclusively for the weekly-strategy replay path.
# They pre-compute a full weekly bar series once per symbol so that the
# inner snapshot loop only needs a cheap cutoff slice instead of a full
# daily→weekly resample on every snapshot.
#
# Why only weekly strategies?
#   Daily-bar strategies (e.g. moving_average) consume the daily slice
#   directly and have no equivalent "resample once, slice many times"
#   opportunity.  If daily-strategy replay ever needs acceleration, it
#   should get its own daily-specific precompute path rather than reusing
#   this weekly cache.
#
# Why this doesn't change completed-week semantics:
#   The full weekly series is built with convert_daily_to_weekly(full_df)
#   — no cutoff applied yet.  The per-snapshot cutoff is applied by
#   slice_weekly_bars_for_snapshot, which mirrors exactly what
#   get_completed_weekly_bars does: it truncates at the last completed
#   Friday anchor for the given snapshot_date.  The strategy therefore
#   sees the same weekly bars it would have seen without the cache.
#
#   With the symbol-level completeness guard (trading_days_count ≥ 3,
#   last_daily_date ≥ last_trading_day), this path is now slightly
#   stricter than get_completed_weekly_bars for extreme short weeks
#   (≤2 trading days).  Only the last bar is validated; earlier
#   historical bars are trusted as-is.
# ---------------------------------------------------------------------------

# Strategies that consume completed weekly bars and can benefit from the
# weekly precompute cache.  Non-weekly strategies are NOT in this set and
# will continue to use the standard daily-slice path.
_WEEKLY_REPLAY_STRATEGIES = frozenset({"momentum_reversal_13", "black_horse"})


def get_replay_strategy_slug(strategy_name: str) -> str:
    """Return the stable replay experiment slug for a strategy name.

    ``momentum_reversal_13`` keeps the historical ``mr13`` slug so existing
    replay checkpoints/results remain resume-compatible. Other strategies use
    their configured strategy name directly.
    """
    legacy_slug_map = {
        "momentum_reversal_13": "mr13",
    }
    return legacy_slug_map.get(strategy_name, strategy_name)


def precompute_weekly_bars_for_replay(full_df: pd.DataFrame) -> pd.DataFrame:
    """Build a complete Friday-anchored weekly bar series from full daily data.

    Called once per symbol before the snapshot loop.  The result is stored
    in weekly_full_cache and sliced per snapshot by
    slice_weekly_bars_for_snapshot.

    No cutoff is applied here — the full history is kept so that any
    snapshot date can be served from this single precomputed series.
    """
    from data_utils import convert_daily_to_weekly
    return convert_daily_to_weekly(full_df, cutoff_date=None)


def slice_weekly_bars_for_snapshot(
    weekly_full: pd.DataFrame,
    snapshot_date: datetime,
    snapshot_week_last_trading_day: object = None,
) -> pd.DataFrame:
    """Return completed weekly bars visible at snapshot_date.

    Mirrors the cutoff logic of get_completed_weekly_bars:
    - snapshot_date carries hour=23:59:59 (set by generate_weekly_snapshot_dates)
    - That means it is always a Friday after 20:00, so the completed-week
      anchor is this Friday itself (not the previous one).
    - We keep all weekly bars whose date <= snapshot_date's Friday anchor.

    Completeness guard (symbol-level):
    The last bar in the sliced result is only kept when it represents a truly
    *completed* trading week.  A bar anchored at the snapshot Friday that was
    built from only Mon-Wed daily data (because the symbol's full_df stopped
    early) is a "half-week" phantom and must be dropped.

    ``snapshot_week_last_trading_day`` is pre-queried once per snapshot week
    (NOT per symbol) and passed in.  When it is None the guard is
    conservative and drops the last bar.

    This is semantically equivalent to calling get_completed_weekly_bars
    on the daily slice, but avoids the resample on every snapshot.

    Equivalence note:
    - This guard is slightly stricter than get_completed_weekly_bars for
      holiday-shortened weeks with 1-2 trading days (threshold = 3).
    - Only the last bar is validated; earlier historical bars are trusted
      as-is from full_df.
    """
    if weekly_full.empty:
        return weekly_full

    # snapshot_date is a Friday at 23:59:59 — the Friday anchor IS snapshot_date.date()
    friday_anchor = pd.Timestamp(snapshot_date.date())
    sliced = weekly_full[weekly_full["date"] <= friday_anchor].copy()

    if sliced.empty:
        return sliced

    has_guard_fields = (
        "trading_days_count" in sliced.columns
        and "last_daily_date" in sliced.columns
    )
    if not has_guard_fields:
        return sliced

    last_bar = sliced.iloc[-1]
    last_bar_date = pd.Timestamp(last_bar["date"])
    if last_bar_date != friday_anchor:
        return sliced

    last_trading_day = snapshot_week_last_trading_day

    if last_trading_day is None:
        sliced = sliced.iloc[:-1]
        return sliced

    last_daily_date = pd.Timestamp(last_bar["last_daily_date"])
    if last_daily_date < pd.Timestamp(last_trading_day):
        sliced = sliced.iloc[:-1]
        return sliced

    trading_days_count = int(last_bar["trading_days_count"])
    if trading_days_count < 3:
        sliced = sliced.iloc[:-1]
        return sliced

    return sliced


def run_weekly_replay_validation(resume: bool = True) -> None:
    """Run weekly historical replay validation for the last lookback_weeks.

    How to Run:
        py -3.13 -c "from scanner_app import run_weekly_replay_validation; run_weekly_replay_validation()"


    Scope:
    - Universe: all A-shares only
    - Time range: most recent lookback_weeks (default 52 weeks)
    - Replay frequency: one run per completed weekly bar
    - Strategy: selected by ``strategy_name`` below

    Output: validation/replay/replay_{universe}_{lookback_weeks}w_{version}_{snapshot_date}.csv
            (per-snapshot result files, overwritten on re-run)
            validation/replay/replay_{universe}_{lookback_weeks}w_{version}_{snapshot_date}_errors.csv
            (per-snapshot error files, overwritten on re-run)
    """
    logger.info("=" * 60)
    logger.info("Starting weekly historical replay validation")
    logger.info("=" * 60)

    ensure_directories()
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "replay"), exist_ok=True)

    stock_pool = "all"
    strategy_name = "momentum_reversal_13"
    strategy_slug = get_replay_strategy_slug(strategy_name)

    universe = stock_pool
    lookback_weeks = 52
    version = "v1"
    experiment_tag = f"{strategy_slug}_{universe}_{lookback_weeks}w_{version}"

    config = load_config()
    strategy_config = {
        "name": strategy_name,
        "params": config.get("strategy", {}).get("params", {})
    }
    data_config = config.get("data", {})

    lookback_days = int(data_config.get("lookback_days", 180))
    initial_days = int(data_config.get("initial_days", 400))
    request_interval = float(data_config.get("request_interval", 0.5))

    required_history_days = lookback_days + lookback_weeks * 7 + 30

    strategy = create_strategy_from_config(strategy_config)

    logger.info("Logging in to baostock...")
    lg = bs.login()
    logger.info("baostock login: %s", lg.error_msg)

    if lg.error_code != "0":
        logger.error("baostock login failed: error_code=%s, error_msg=%s", lg.error_code, lg.error_msg)
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    try:
        stocks = get_stock_list(stock_pool)
        logger.info("Processing %s symbols for weekly replay", len(stocks))

        weekly_dates = generate_weekly_snapshot_dates(lookback_weeks)
        logger.info("Generated %s weekly snapshot dates", len(weekly_dates))

        # Build the snapshot window identity once: this list is the
        # authoritative window for the current run and is stored in the
        # checkpoint so future resumes can verify they target the same window.
        snapshot_dates_str = [d.strftime("%Y-%m-%d") for d in weekly_dates]

        # Calculate fixed replay_data_end_date: the data boundary for this replay experiment.
        # This ensures the same data is used regardless of when the replay is resumed.
        # Use last snapshot + 21 weeks (20 weeks max forward return + 1 week buffer).
        last_snapshot_date = max(weekly_dates).date()
        replay_data_end_date = (last_snapshot_date + timedelta(weeks=21)).strftime("%Y-%m-%d")
        logger.info("Replay data end date (fixed): %s", replay_data_end_date)

        checkpoint_path = get_replay_checkpoint_path(universe, lookback_weeks, version)

        completed_snapshots: List[str] = []
        failed_symbols_per_snapshot: Dict[str, List[str]] = {}
        run_mode = "fresh"
        resume_replay_data_end_date = None

        if resume and os.path.exists(checkpoint_path):
            logger.info("[Replay] Checkpoint detected: %s", checkpoint_path)
            checkpoint = load_replay_checkpoint(checkpoint_path)

            # Validate checkpoint and get completed snapshots
            completed_snapshots = validate_replay_checkpoint(
                checkpoint,
                experiment_tag,
                universe,
                lookback_weeks,
                version,
                snapshot_dates_str,
                expected_replay_data_end_date=replay_data_end_date,
            )

            # Use the replay_data_end_date from checkpoint for resume
            resume_replay_data_end_date = checkpoint.get("replay_data_end_date")
            if resume_replay_data_end_date is None:
                raise RuntimeError(
                    "Replay checkpoint is missing 'replay_data_end_date'. "
                    "The checkpoint schema is insufficient to verify that the "
                    "data boundary is identical to the current run. "
                    "Cannot safely resume — please delete the checkpoint or "
                    "run a fresh replay with the new schema."
                )
            # For resume, use the checkpoint's value to ensure consistency
            replay_data_end_date = resume_replay_data_end_date
            logger.info("[Replay] Resume with replay_data_end_date: %s", replay_data_end_date)

            # Verify checkpoint consistency: all completed snapshots must have their result files
            _verify_completed_snapshot_files(
                completed_snapshots, universe, lookback_weeks, version
            )

            # Extract and log any unresolved failed symbols from previous runs.
            # These are quality/omission records — they do NOT block resume.
            failed_symbols_per_snapshot = checkpoint.get("failed_symbols_per_snapshot", {})
            if failed_symbols_per_snapshot:
                summary = _summarize_failed_symbols(failed_symbols_per_snapshot)
                logger.warning(summary)

            run_mode = "resume"
        elif resume:
            logger.info("[Replay] No checkpoint detected; starting fresh replay run")

        if run_mode == "fresh":
            # Conservative: refuse to silently destroy existing per-snapshot output files
            # when there is no checkpoint. The checkpoint is the sole source of truth;
            # per-snapshot CSV files are just result carriers and must not be
            # used to infer completion status or overwrite intent.
            replay_dir = os.path.join(VALIDATION_DIR, "replay")
            if os.path.exists(replay_dir):
                import glob
                pattern = os.path.join(
                    replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_????-??-??*.csv"
                )
                existing_files = glob.glob(pattern)
                if existing_files:
                    raise RuntimeError(
                        "No replay checkpoint found, but existing per-snapshot replay files "
                        "are already present.  The checkpoint is the sole source of truth "
                        "for completed snapshots; per-snapshot CSV files are result carriers only "
                        "and cannot be used to infer progress or overwrite intent.  "
                        "Refusing to proceed to avoid inconsistent state.  "
                        "Please manually remove or relocate the existing output file(s) "
                        "before running the replay again."
                    )
            # Remove any stale checkpoint to start clean
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        completed_snapshot_set = set(completed_snapshots)
        snapshots_to_process = [
            snapshot_date for snapshot_date in weekly_dates
            if snapshot_date.strftime("%Y-%m-%d") not in completed_snapshot_set
        ]

        logger.info(
            "[Replay] Run mode=%s completed_snapshots=%d skipped=%d pending=%d",
            run_mode,
            len(completed_snapshots),
            len(weekly_dates) - len(snapshots_to_process),
            len(snapshots_to_process),
        )

        if run_mode == "fresh":
            # For fresh run, remove any existing checkpoint to start clean
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        full_df_cache: Dict[str, pd.DataFrame | None] = {}
        # Weekly-strategy replay optimization v1:
        # Pre-computed full weekly bar series, keyed by symbol.
        # Built once per symbol from full_df; each snapshot slices it cheaply.
        # Only populated when the strategy is in _WEEKLY_REPLAY_STRATEGIES.
        # Daily-strategy replay does not use this cache.
        weekly_full_cache: Dict[str, pd.DataFrame | None] = {}
        use_weekly_cache = strategy.name in _WEEKLY_REPLAY_STRATEGIES
        SLOW_SYMBOL_THRESHOLD = 3.0  # seconds — warn if a single symbol exceeds this

        for snapshot_idx, snapshot_date in enumerate(weekly_dates, 1):
            snapshot_start = time.time()
            snapshot_date_str = snapshot_date.strftime("%Y-%m-%d")

            if snapshot_date_str in completed_snapshot_set:
                logger.info(
                    "[Replay] Snapshot %d/%d skipped via checkpoint: date=%s",
                    snapshot_idx, len(weekly_dates), snapshot_date_str,
                )
                continue

            logger.info(
                "[Replay] Snapshot %d/%d started: date=%s",
                snapshot_idx, len(weekly_dates), snapshot_date_str,
            )

            context = StrategyContext(
                now=snapshot_date,
                stock_pool=stock_pool,
                config=config
            )

            decision = strategy.can_run(context)
            if not decision.should_run:
                logger.debug("Strategy skipped for %s: %s",
                           snapshot_date_str, decision.reason_text)

                # This snapshot has been fully evaluated: the snapshot-level
                # decision is complete.  Mark it as completed so that resume
                # will skip it on the next run.
                snapshot_elapsed = time.time() - snapshot_start
                logger.info(
                    "[Replay] Snapshot %d/%d done (no-op): date=%s elapsed=%.1fs "
                    "processed=0 matched=0 failed=0",
                    snapshot_idx, len(weekly_dates), snapshot_date_str, snapshot_elapsed,
                )

                # Write result carrier file first (even if empty) before checkpoint
                write_replay_results(
                    [], universe, lookback_weeks, version, snapshot_date_str
                )

                # Clean up any stale error file for this no-op snapshot
                write_replay_errors(
                    [], universe, lookback_weeks, version, snapshot_date_str
                )

                # Now mark as completed and save checkpoint
                completed_snapshot_set.add(snapshot_date_str)
                completed_snapshots.append(snapshot_date_str)
                completed_snapshots.sort()
                failed_symbols_per_snapshot.pop(snapshot_date_str, None)
                save_replay_checkpoint(
                    checkpoint_path, experiment_tag, universe, lookback_weeks, version,
                    completed_snapshots.copy(), snapshot_dates_str,
                    replay_data_end_date=replay_data_end_date,
                    failed_symbols_per_snapshot=failed_symbols_per_snapshot,
                )
                continue

            snapshot_results = []
            processed_count = 0
            failed_count = 0
            snapshot_errors = []
            snapshot_failed_symbols: List[str] = []

            snapshot_week_last_trading_day = None
            if use_weekly_cache:
                snapshot_week_last_trading_day = get_last_trading_day_of_week(
                    pd.Timestamp(snapshot_date.date())
                )
                if snapshot_week_last_trading_day is None:
                    logger.warning(
                        "[Replay] Snapshot %d/%d date=%s: trading calendar query failed, "
                        "snapshot-week last bar will be dropped conservatively",
                        snapshot_idx, len(weekly_dates), snapshot_date_str,
                    )

            for symbol_idx, symbol in enumerate(stocks, 1):
                symbol_start = time.time()
                stage_times: Dict[str, float] = {}
                exit_stage = ""
                matched = False
                symbol_failed = False
                error_record = None
                current_stage = "unknown"

                try:
                    # Stage: load_full_df_for_replay
                    if symbol not in full_df_cache:
                        t0 = time.time()
                        current_stage = "load_full_df"
                        full_df_cache[symbol] = load_full_df_for_replay(
                            symbol, lookback_days, initial_days, request_interval, required_history_days,
                            replay_data_end_date=replay_data_end_date,
                        )
                        stage_times["load_full_df"] = time.time() - t0

                        # Weekly-strategy replay optimization v1:
                        # Pre-compute full weekly bars once per symbol so that
                        # each snapshot only needs a cheap cutoff slice.
                        # This is the weekly-specific acceleration path; daily
                        # strategies do not populate weekly_full_cache.
                        if use_weekly_cache and full_df_cache[symbol] is not None:
                            weekly_full_cache[symbol] = precompute_weekly_bars_for_replay(
                                full_df_cache[symbol]
                            )

                    full_df = full_df_cache[symbol]

                    if full_df is None or full_df.empty:
                        exit_stage = "full_df_empty"

                    if not exit_stage:
                        # Stage: load_historical_data_up_to_date
                        t0 = time.time()
                        current_stage = "load_historical"
                        df = load_historical_data_up_to_date(
                            symbol, snapshot_date, lookback_days, initial_days, request_interval, full_df=full_df
                        )
                        stage_times["load_historical"] = time.time() - t0

                        if df is None or df.empty:
                            exit_stage = "hist_empty"

                    if not exit_stage:
                        processed_count += 1

                        # Stage: strategy.scan
                        # Weekly-strategy replay optimization v1:
                        # For weekly strategies, pass the pre-sliced weekly bars
                        # so the strategy skips the daily→weekly resample.
                        # Non-weekly strategies use the standard call path.
                        t0 = time.time()
                        current_stage = "scan"
                        if use_weekly_cache and symbol in weekly_full_cache and weekly_full_cache[symbol] is not None:
                            precomputed_weekly = slice_weekly_bars_for_snapshot(
                                weekly_full_cache[symbol], snapshot_date,
                                snapshot_week_last_trading_day,
                            )
                            result = strategy.scan(symbol, df, context, precomputed_weekly=precomputed_weekly)
                        else:
                            result = strategy.scan(symbol, df, context)
                        stage_times["scan"] = time.time() - t0
                        matched = result.matched

                        # Stage: calculate_forward_returns (matched only)
                        if matched:
                            t0 = time.time()
                            current_stage = "calc_returns"
                            forward_returns = calculate_forward_returns(symbol, result, full_df=full_df)
                            stage_times["calc_returns"] = time.time() - t0

                            current_stage = "build_record"
                            replay_record = build_replay_record(
                                symbol, result, context, forward_returns,
                                experiment_tag, universe, lookback_weeks
                            )
                            snapshot_results.append(replay_record)

                except Exception as exc:
                    symbol_failed = True
                    failed_count += 1
                    snapshot_failed_symbols.append(symbol)

                    # Log error
                    logger.error(
                        "[Replay] Snapshot %d/%d date=%s symbol %s failed at stage %s: %s: %s",
                        snapshot_idx, len(weekly_dates), snapshot_date_str, symbol, current_stage,
                        exc.__class__.__name__, str(exc)
                    )

                    # Create error record
                    error_record = {
                        "experiment_tag": experiment_tag,
                        "snapshot_date": snapshot_date_str,
                        "snapshot_index": snapshot_idx,
                        "code": symbol,
                        "stage": current_stage,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc)
                    }
                    snapshot_errors.append(error_record)

                # -- Unified symbol-level observability 收口 --
                symbol_elapsed = time.time() - symbol_start

                if symbol_idx % 100 == 0 or matched or symbol_failed:
                    log_msg = (
                        f"[Replay] Snapshot {snapshot_idx}/{len(weekly_dates)} "
                        f"symbol {symbol_idx}/{len(stocks)} {symbol} | "
                        f"matched={matched} elapsed={symbol_elapsed:.2f}s"
                    )
                    if "scan" in stage_times:
                        log_msg += f" scan={stage_times['scan']:.2f}s"
                    if exit_stage:
                        log_msg += f" exit={exit_stage}"
                    if symbol_failed:
                        log_msg += " FAILED"
                    logger.info(log_msg)

                if symbol_elapsed > SLOW_SYMBOL_THRESHOLD:
                    stage_detail = ", ".join(
                        f"{k}={v:.2f}s" for k, v in stage_times.items()
                    )
                    warning_msg = (
                        f"[Replay] Slow symbol {symbol} in snapshot {snapshot_idx} ({snapshot_date_str}): "
                        f"total={symbol_elapsed:.2f}s [{stage_detail}]"
                    )
                    if exit_stage:
                        warning_msg += f" exit={exit_stage}"
                    if symbol_failed:
                        warning_msg += " FAILED"
                    logger.warning(warning_msg)

            snapshot_elapsed = time.time() - snapshot_start

            # Write per-snapshot result file (even if empty — ensures carrier file exists)
            write_replay_results(
                snapshot_results, universe, lookback_weeks, version, snapshot_date_str
            )

            # Write per-snapshot error file (overwrite mode; removes stale file if no errors)
            write_replay_errors(
                snapshot_errors, universe, lookback_weeks, version, snapshot_date_str
            )

            # Only after files are written successfully, mark snapshot as completed
            completed_snapshot_set.add(snapshot_date_str)
            completed_snapshots.append(snapshot_date_str)
            completed_snapshots.sort()

            # Update failed_symbols_per_snapshot: record failed symbols or clear stale entry
            if snapshot_failed_symbols:
                failed_symbols_per_snapshot[snapshot_date_str] = sorted(
                    list(set(snapshot_failed_symbols))
                )
            else:
                failed_symbols_per_snapshot.pop(snapshot_date_str, None)

            logger.info(
                "[Replay] Snapshot %d/%d done: date=%s elapsed=%.1fs "
                "processed=%d matched=%d failed=%d",
                snapshot_idx, len(weekly_dates), snapshot_date_str,
                snapshot_elapsed, processed_count, len(snapshot_results), failed_count,
            )

            # Save checkpoint after successful file write
            save_replay_checkpoint(
                checkpoint_path,
                experiment_tag,
                universe,
                lookback_weeks,
                version,
                completed_snapshots.copy(),
                snapshot_dates_str,
                replay_data_end_date=replay_data_end_date,
                failed_symbols_per_snapshot=failed_symbols_per_snapshot,
            )

        # Log unresolved failure summary at replay end
        if failed_symbols_per_snapshot:
            summary = _summarize_failed_symbols(failed_symbols_per_snapshot)
            logger.warning(summary)

        logger.info("Weekly replay validation completed.")

    finally:
        bs.logout()
        logger.info("Weekly replay validation finished")


def merge_replay_snapshot_files(
    universe: str,
    lookback_weeks: int,
    version: str,
    output_path: str | None = None,
) -> str:
    """Merge per-snapshot result files into a single CSV.

    This is a helper function for downstream analysis. It reads all
    per-snapshot result files matching the pattern and merges them
    into a single CSV file.

    Args:
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        output_path: Output file path. If None, defaults to the old aggregate name.

    Returns:
        Path to the merged CSV file.
    """
    import glob

    replay_dir = os.path.join(VALIDATION_DIR, "replay")
    pattern = os.path.join(
        replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_????-??-??.csv"
    )

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No per-snapshot result files found matching: {pattern}")

    logger.info("Merging %d per-snapshot result files...", len(files))

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    if output_path is None:
        output_path = os.path.join(
            replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_merged.csv"
        )

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Merged result written to %s (%d rows)", output_path, len(merged))

    return output_path


def merge_replay_error_snapshot_files(
    universe: str,
    lookback_weeks: int,
    version: str,
    output_path: str | None = None,
) -> str:
    """Merge per-snapshot error files into a single CSV.

    This is a helper function for downstream analysis. It reads all
    per-snapshot error files matching the pattern and merges them
    into a single CSV file.

    Args:
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        output_path: Output file path. If None, defaults to the old aggregate name.

    Returns:
        Path to the merged CSV file.
    """
    import glob

    replay_dir = os.path.join(VALIDATION_DIR, "replay")
    pattern = os.path.join(
        replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_????-??-??_errors.csv"
    )

    files = sorted(glob.glob(pattern))
    if not files:
        logger.info("No per-snapshot error files found matching: %s", pattern)
        output_path = output_path or os.path.join(
            replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_errors_merged.csv"
        )
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    logger.info("Merging %d per-snapshot error files...", len(files))

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    if output_path is None:
        output_path = os.path.join(
            replay_dir, f"replay_{universe}_{lookback_weeks}w_{version}_errors_merged.csv"
        )

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Merged errors written to %s (%d rows)", output_path, len(merged))

    return output_path


if __name__ == "__main__":
    main()
