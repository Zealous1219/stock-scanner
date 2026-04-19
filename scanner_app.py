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


def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        rs = getattr(bs, api_name)(day=datetime.now().strftime("%Y-%m-%d"))
    else:
        rs = getattr(bs, api_name)()

    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError(f"Failed to fetch stock list for pool {stock_pool}")

    df = pd.DataFrame(data_list, columns=rs.fields)
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
        if latest_date >= yesterday:
            return local_df.tail(lookback_days), performed_remote_fetch

        new_data = fetch_historical_data(symbol, latest_date.strftime("%Y-%m-%d"), today)
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
        strategy_config = {**strategy_config, "name": env_strategy_name}
        logger.info("Using strategy override from STRATEGY_NAME=%s", env_strategy_name)
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

    lg = bs.login()
    logger.info("baostock login: %s", lg.error_msg)

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
