"""Config-driven stock scanner application."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import baostock as bs
import pandas as pd

from config_loader import load_config
from data_utils import ensure_daily_frame
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
) -> pd.DataFrame | None:
    """Load complete, standardized full_df for a stock for replay.

    Returns DataFrame with datetime date column, or None if load failed.
    """
    filename = symbol.replace(".", "_")
    file_path = os.path.join(DATA_DIR, f"{filename}.csv")

    history_days = max(initial_days, required_history_days)
    end_date = datetime.now().strftime("%Y-%m-%d")
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

        earliest_date = local_df["date"].min()
        required_start = pd.Timestamp(datetime.now() - timedelta(days=history_days))

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
        today_timestamp = pd.Timestamp(datetime.now().date())
        today_str = datetime.now().strftime("%Y-%m-%d")

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
    """Generate weekly Friday dates for the last lookback_weeks.

    Args:
        lookback_weeks: Number of weeks to look back from current date
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=lookback_weeks)

    friday_dates = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.weekday() == 4:
            friday_dates.append(current_date)
        current_date += timedelta(days=1)

    friday_dates = [d for d in friday_dates if d <= end_date]
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
    }

    signal_date_str = result.details.get("signal_date", "")
    if not signal_date_str:
        return forward_returns

    try:
        signal_date = pd.to_datetime(signal_date_str).date()

        future_dates = {
            "4w": signal_date + timedelta(weeks=4),
            "8w": signal_date + timedelta(weeks=8),
            "12w": signal_date + timedelta(weeks=12),
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
            future_data = df[df["date"] >= pd.Timestamp(future_date)]
            if future_data.empty:
                continue

            future_row = future_data.iloc[0]
            future_close = float(future_row["close"])

            returns = (future_close - signal_close) / signal_close

            if period == "4w":
                forward_returns["return_4w"] = returns
            elif period == "8w":
                forward_returns["return_8w"] = returns
            elif period == "12w":
                forward_returns["return_12w"] = returns

    except Exception as exc:
        logger.debug("计算%s前向收益率失败: %s", symbol, exc)

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
    }

    return record


def write_replay_results(results: List[Dict[str, Any]], universe: str, lookback_weeks: int, version: str, write_header: bool = True) -> str:
    """Write replay validation results to CSV.

    Args:
        results: List of replay records to write
        universe: Stock universe identifier
        lookback_weeks: Number of weeks looked back
        version: Version identifier
        write_header: Whether to write CSV header (True for new file, False for append)
    """
    replay_file = os.path.join(VALIDATION_DIR, "replay", f"replay_{universe}_{lookback_weeks}w_{version}.csv")

    df = pd.DataFrame(results)

    if write_header:
        df.to_csv(replay_file, index=False, encoding="utf-8-sig")
        logger.info("Replay results written to %s (with header)", replay_file)
    else:
        df.to_csv(replay_file, mode="a", header=False, index=False, encoding="utf-8-sig")
        logger.info("Replay results appended to %s (no header)", replay_file)

    return replay_file


def run_weekly_replay_validation() -> None:
    """Run weekly historical replay validation for the last lookback_weeks.

    How to Run:
        py -3.13 -c "from scanner_app import run_weekly_replay_validation; run_weekly_replay_validation()"


    Scope:
    - Universe: all A-shares only
    - Time range: most recent lookback_weeks (default 52 weeks)
    - Replay frequency: one run per completed weekly bar
    - Strategy: momentum_reversal_13 only

    Output: validation/replay/replay_{universe}_{lookback_weeks}w_{version}.csv
    """
    logger.info("=" * 60)
    logger.info("Starting weekly historical replay validation")
    logger.info("=" * 60)

    ensure_directories()
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "replay"), exist_ok=True)

    stock_pool = "all"
    strategy_name = "momentum_reversal_13"

    universe = stock_pool
    lookback_weeks = 52
    version = "v1"
    experiment_tag = f"mr13_{universe}_{lookback_weeks}w_{version}"

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

        replay_file = os.path.join(VALIDATION_DIR, "replay", f"replay_{universe}_{lookback_weeks}w_{version}.csv")

        if os.path.exists(replay_file):
            logger.info("Clearing existing replay file: %s", replay_file)
            open(replay_file, 'w').close()
        else:
            logger.info("Creating new replay file: %s", replay_file)

        full_df_cache: Dict[str, pd.DataFrame | None] = {}

        for snapshot_idx, snapshot_date in enumerate(weekly_dates, 1):
            logger.info("Processing snapshot %s/%s: %s",
                       snapshot_idx, len(weekly_dates), snapshot_date.strftime("%Y-%m-%d"))

            context = StrategyContext(
                now=snapshot_date,
                stock_pool=stock_pool,
                config=config
            )

            decision = strategy.can_run(context)
            if not decision.should_run:
                logger.debug("Strategy skipped for %s: %s",
                           snapshot_date.strftime("%Y-%m-%d"), decision.reason_text)
                continue

            snapshot_results = []

            for symbol_idx, symbol in enumerate(stocks, 1):
                if symbol_idx % 100 == 0:
                    logger.info("  Progress %s/%s for snapshot %s/%s: %s",
                               symbol_idx, len(stocks), snapshot_idx, len(weekly_dates), snapshot_date.strftime("%Y-%m-%d"))

                if symbol not in full_df_cache:
                    full_df_cache[symbol] = load_full_df_for_replay(
                        symbol, lookback_days, initial_days, request_interval, required_history_days
                    )

                full_df = full_df_cache[symbol]

                if full_df is None or full_df.empty:
                    continue

                df = load_historical_data_up_to_date(
                    symbol, snapshot_date, lookback_days, initial_days, request_interval, full_df=full_df
                )

                if df is None or df.empty:
                    continue

                result = strategy.scan(symbol, df, context)

                if result.matched:
                    forward_returns = calculate_forward_returns(symbol, result, full_df=full_df)

                    replay_record = build_replay_record(
                        symbol, result, context, forward_returns,
                        experiment_tag, universe, lookback_weeks
                    )
                    snapshot_results.append(replay_record)

            if snapshot_results:
                write_header = (snapshot_idx == 1) or (os.path.exists(replay_file) and os.path.getsize(replay_file) == 0)
                write_replay_results(snapshot_results, universe, lookback_weeks, version, write_header=write_header)
                logger.info("Snapshot %s/%s completed: %s signals written",
                           snapshot_idx, len(weekly_dates), len(snapshot_results))
            else:
                logger.info("Snapshot %s/%s completed: no signals found",
                           snapshot_idx, len(weekly_dates))

        logger.info("Weekly replay validation completed.")

    finally:
        bs.logout()
        logger.info("Weekly replay validation finished")


if __name__ == "__main__":
    main()
