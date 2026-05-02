"""Config-driven stock scanner application. 入口及 replay 验证框架。"""

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
from data_utils import ensure_daily_frame, get_last_trading_day_of_week, get_snapshot_trading_week_info
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
    """创建 data/ 和 output/ 目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def should_force_refresh_on_friday(now: datetime) -> bool:
    """周五 20:00 后强制刷新数据。"""
    return now.weekday() == 4 and now.hour >= 20


def get_latest_trading_day(reference_time: datetime | None = None, lookback_days: int = 14) -> str:
    """返回参考日期前最近的一个交易日（含参考日期）。"""
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
    """加载或刷新 symbol 日线缓存。按需远程拉取。

    Returns: (df, performed_remote_fetch)
    """
    filename = symbol.replace(".", "_")
    file_path = os.path.join(DATA_DIR, f"{filename}.csv")

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=initial_days)).strftime("%Y-%m-%d")
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
        today_date = now.date()
        yesterday = pd.Timestamp(today_date) - pd.Timedelta(days=1)
        today_timestamp = pd.Timestamp(today_date)
        today_str = today

        force_refresh = should_force_refresh_on_friday(now)

        if force_refresh:
            # 周五 20:00 后，如果已有当日数据则直接用缓存
            if latest_date >= today_timestamp:
                return local_df.tail(lookback_days), performed_remote_fetch
        elif latest_date >= yesterday:
            # 非周末强制刷新，用正常缓存逻辑
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
        # STRATEGY_NAME 环境变量覆盖：从目标策略类读取 DEFAULT_PARAMS，避免旧参数污染
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
    """加载 replay 用的完整日线数据（比 lookback_days 更长的历史）。

    Args:
        replay_data_end_date: 固定数据边界 YYYY-MM-DD，确保 replay 可复现。
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
    """生成过去 N 周（默认 52 周）的周五 snapshot 日期列表。

    每个 snapshot 为周五 23:59:59，保证 replay 中 get_last_completed_week_end
    看到 hour≥20 从而一致地将本周判定为"已完成"。

    通过从最后完成的周五递减生成，确保总是返回恰好 N 个 snapshot。
    """
    now = datetime.now()
    today = now.date()
    weekday = now.weekday()

    # 1. 确定本周的 Friday 锚点
    if weekday < 4:
        this_friday_anchor = today + timedelta(days=4 - weekday)
    elif weekday == 4:
        this_friday_anchor = today
    else:
        this_friday_anchor = today - timedelta(days=weekday - 4)

    # 2. 查询本交易周最后一个交易日
    last_trading_day = None
    try:
        last_trading_day = get_last_trading_day_of_week(pd.Timestamp(today))
    except Exception:
        pass

    # 3. 判断本周是否已完成
    if last_trading_day is None:
        last_completed_friday = this_friday_anchor - timedelta(weeks=1)
    else:
        last_trading_date = last_trading_day.date()
        if today < last_trading_date:
            last_completed_friday = this_friday_anchor - timedelta(weeks=1)
        elif today > last_trading_date:
            last_completed_friday = this_friday_anchor
        else:
            if now.hour < 20:
                last_completed_friday = this_friday_anchor - timedelta(weeks=1)
            else:
                last_completed_friday = this_friday_anchor

    # 从最后完成的周五向前递减，生成恰好 N 个周五日期
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
    """加载截止至 cutoff_date 的历史日线数据。

    如果提供了 full_df，则直接从内存过滤（replay 优化路径），不从磁盘读取。
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
    """计算信号日之后的 4w/8w/12w/16w/20w 前向收益率。

    若目标日期无交易，在 3 日内查找最近交易日；超过 3 日则返回 NaN。
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
        raise

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
    """构建 replay 输出记录：核心信号字段 + 前向收益率。"""
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


# Replay 输出 CSV 列定义（需与 build_replay_record 保持一致）
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
    strategy_slug: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    snapshot_date: str,
) -> str:
    """写入 per-snapshot replay 结果 CSV。

    覆盖写入（非追加），即使结果为空也写出表头保持 schema 稳定。
    """
    replay_file = get_replay_snapshot_result_path(strategy_slug, universe, lookback_weeks, version, snapshot_date)

    if results:
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame(columns=_REPLAY_RESULT_COLUMNS)
    df.to_csv(replay_file, index=False, encoding="utf-8-sig")
    logger.info("Replay results written to %s (%d rows)", replay_file, len(df))
    return replay_file


def write_replay_errors(
    errors: List[Dict[str, Any]],
    strategy_slug: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    snapshot_date: str,
) -> str:
    """写入 per-snapshot replay 错误 CSV。

    覆盖写入，错误列表为空时删除旧错误文件（防残留）。
    """
    error_file = get_replay_snapshot_error_path(strategy_slug, universe, lookback_weeks, version, snapshot_date)

    if not errors:
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


def get_replay_snapshot_result_path(strategy_slug: str, universe: str, lookback_weeks: int, version: str, snapshot_date: str) -> str:
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_{snapshot_date}.csv",
    )


def get_replay_snapshot_error_path(strategy_slug: str, universe: str, lookback_weeks: int, version: str, snapshot_date: str) -> str:
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_{snapshot_date}_errors.csv",
    )


def get_replay_checkpoint_path(strategy_slug: str, universe: str, lookback_weeks: int, version: str) -> str:
    return os.path.join(
        VALIDATION_DIR,
        "replay",
        f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}.checkpoint.json",
    )


def _verify_completed_snapshot_files(
    completed_snapshots: List[str],
    strategy_slug: str,
    universe: str,
    lookback_weeks: int,
    version: str,
) -> None:
    """校验 checkpoint 中标记为 completed 的 snapshot 都有对应的结果文件。

    任一缺失即拒绝 resume（防不一致状态）。
    """
    missing_files = []

    for snapshot_date in completed_snapshots:
        result_path = get_replay_snapshot_result_path(
            strategy_slug, universe, lookback_weeks, version, snapshot_date
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
    """生成可读的失败 symbol 摘要日志（含前 5 个 snapshot 样例）。"""
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
    """从 JSON 加载 replay checkpoint。"""
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
    """校验 checkpoint 身份一致性，返回已完成的 snapshot 列表。

    Checkpoint 是进度的唯一权威来源，CSV 输出仅作为结果载体不可用来推断进度。

    校验项：
      - experiment_tag / universe / lookback_weeks / version
      - snapshot_dates（完整快照窗口一致性）
      - replay_data_end_date（数据边界一致性）
      - completed_snapshots 类型/格式
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

    # 校验 snapshot_dates：必须与当前运行完全一致
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

    # 校验 replay_data_end_date：数据边界必须一致
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

    # 校验 failed_symbols_per_snapshot（可选字段，仅做质量记录不做进度阻断）
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
    """持久化 replay checkpoint，支持 snapshot 级 resume。

    - snapshot_dates: 编码完整快照窗口身份，用于 resume 时校验
    - replay_data_end_date: 固定数据边界，保证 resume 时使用相同数据
    - failed_symbols_per_snapshot: 失败 symbol 的记录，仅做质量观测不做进度阻断
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
# 周线策略 replay 加速专用路径：预先从 full_df 计算全量周线，每个 snapshot
# 只需做廉价切片，避免对每个 snapshot 重复 reample。
#
# 适用策略：momentum_reversal_13, black_horse（在 _WEEKLY_REPLAY_STRATEGIES 中声明）
# 日线策略（如 moving_average）不走此路径。
#
# 等价性保证：
#   precompute = convert_daily_to_weekly(full_df, cutoff_date=None)  — 不截断
#   slice_weekly_bars_for_snapshot 用 date <= friday_anchor 截断
#   结果等同于 get_completed_weekly_bars 在对应 snapshot 时刻的输出。
#
# 符号级完成守卫：
#   _guard_snapshot_week_completion 对 snapshot 边界处的末根 bar 做校验：
#   - last_daily_date < last_trading_day → 丢弃（数据不完整）
#   - last_trading_day 为 None → 保守丢弃（日历查询失败）
#   仅校验末根 bar；历史 bar 信任 full_df 的原始数据。
# ---------------------------------------------------------------------------

_WEEKLY_REPLAY_STRATEGIES = frozenset({"momentum_reversal_13", "black_horse"})


def get_replay_strategy_slug(strategy_name: str) -> str:
    """返回 replay 实验的策略标识 slug。

    momentum_reversal_13 → mr13（保持与历史 checkpoint 兼容）
    其他策略使用策略名本身。
    """
    legacy_slug_map = {
        "momentum_reversal_13": "mr13",
    }
    return legacy_slug_map.get(strategy_name, strategy_name)


def precompute_weekly_bars_for_replay(full_df: pd.DataFrame) -> pd.DataFrame:
    """从全量日线预计算 W-FRI 周线序列（无截断）。"""
    from data_utils import convert_daily_to_weekly
    return convert_daily_to_weekly(full_df, cutoff_date=None)


def _guard_snapshot_week_completion(
    weekly_bars: pd.DataFrame,
    friday_anchor: pd.Timestamp,
    snapshot_week_last_trading_day: object,
) -> pd.DataFrame:
    """保守守卫：日历查询失败或数据不完整时丢弃 snapshot 周的末根 bar。

    统一主路径与 fallback 路径在 calendar 异常时的语义。仅检查末根 bar，
    历史周线不受影响。
    """
    if weekly_bars.empty:
        return weekly_bars
    has_guard_fields = (
        "trading_days_count" in weekly_bars.columns
        and "last_daily_date" in weekly_bars.columns
    )
    if not has_guard_fields:
        return weekly_bars
    last_bar = weekly_bars.iloc[-1]
    last_bar_date = pd.Timestamp(last_bar["date"])
    if last_bar_date != friday_anchor:
        return weekly_bars
    last_trading_day = snapshot_week_last_trading_day
    if last_trading_day is None:
        return weekly_bars.iloc[:-1]
    last_daily_date = pd.Timestamp(last_bar["last_daily_date"])
    if last_daily_date < pd.Timestamp(last_trading_day):
        return weekly_bars.iloc[:-1]
    trading_days_count = int(last_bar["trading_days_count"])
    if trading_days_count < 3:
        return weekly_bars.iloc[:-1]
    return weekly_bars


def slice_weekly_bars_for_snapshot(
    weekly_full: pd.DataFrame,
    snapshot_date: datetime,
    snapshot_week_last_trading_day: object = None,
) -> pd.DataFrame:
    """返回 snapshot 时刻可见的已完成周线。

    等价于在对应 snapshot 调用 get_completed_weekly_bars，
    但通过预计算避免重复 reample。

    符号级守卫：snapshot_week_last_trading_day 为 None 时保守丢弃末根 bar。
    """
    if weekly_full.empty:
        return weekly_full

    friday_anchor = pd.Timestamp(snapshot_date.date())
    sliced = weekly_full[weekly_full["date"] <= friday_anchor].copy()

    if sliced.empty:
        return sliced

    return _guard_snapshot_week_completion(sliced, friday_anchor, snapshot_week_last_trading_day)


def run_weekly_replay_validation(resume: bool = True) -> None:
    """运行周级历史 replay 验证。

    用法：
        py -3.13 -c "from scanner_app import run_weekly_replay_validation; run_weekly_replay_validation()"

    范围：
      - 股票池：全 A 股
      - 时间：最近 lookback_weeks 个 snapshot（默认 52 周）
      - 策略：由 config.json 中 strategy.name 指定

    输出：
      validation/replay/replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_{snapshot_date}.csv
      validation/replay/replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_{snapshot_date}_errors.csv
    """
    logger.info("=" * 60)
    logger.info("Starting weekly historical replay validation")
    logger.info("=" * 60)

    ensure_directories()
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "replay"), exist_ok=True)

    config = load_config()
    strategy_name = config.get("strategy", {}).get("name") or "momentum_reversal_13"
    strategy_slug = get_replay_strategy_slug(strategy_name)

    logger.info("Replay strategy: name=%s, slug=%s", strategy_name, strategy_slug)

    stock_pool = "all"
    universe = stock_pool
    lookback_weeks = 52
    version = "v1"
    experiment_tag = f"{strategy_slug}_{universe}_{lookback_weeks}w_{version}"

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

        # 构建一次性快照窗口标识，写入 checkpoint 供 resume 校验
        snapshot_dates_str = [d.strftime("%Y-%m-%d") for d in weekly_dates]

        # 计算固定 replay 数据边界：最后 snapshot + 21 周（20 周最大 forward return + 1 周 buffer）
        last_snapshot_date = max(weekly_dates).date()
        replay_data_end_date = (last_snapshot_date + timedelta(weeks=21)).strftime("%Y-%m-%d")
        logger.info("Replay data end date (fixed): %s", replay_data_end_date)

        checkpoint_path = get_replay_checkpoint_path(strategy_slug, universe, lookback_weeks, version)

        completed_snapshots: List[str] = []
        failed_symbols_per_snapshot: Dict[str, List[str]] = {}
        run_mode = "fresh"
        resume_replay_data_end_date = None

        if resume and os.path.exists(checkpoint_path):
            logger.info("[Replay] Checkpoint detected: %s", checkpoint_path)
            checkpoint = load_replay_checkpoint(checkpoint_path)

            completed_snapshots = validate_replay_checkpoint(
                checkpoint,
                experiment_tag,
                universe,
                lookback_weeks,
                version,
                snapshot_dates_str,
                expected_replay_data_end_date=replay_data_end_date,
            )

            resume_replay_data_end_date = checkpoint.get("replay_data_end_date")
            if resume_replay_data_end_date is None:
                raise RuntimeError(
                    "Replay checkpoint is missing 'replay_data_end_date'. "
                    "The checkpoint schema is insufficient to verify that the "
                    "data boundary is identical to the current run. "
                    "Cannot safely resume — please delete the checkpoint or "
                    "run a fresh replay with the new schema."
                )
            replay_data_end_date = resume_replay_data_end_date
            logger.info("[Replay] Resume with replay_data_end_date: %s", replay_data_end_date)

            # 校验：所有已完成的 snapshot 必须有对应结果文件
            _verify_completed_snapshot_files(
                completed_snapshots, strategy_slug, universe, lookback_weeks, version
            )

            # 输出之前未解决的失败 symbol 摘要（不影响 resume）
            failed_symbols_per_snapshot = checkpoint.get("failed_symbols_per_snapshot", {})
            if failed_symbols_per_snapshot:
                summary = _summarize_failed_symbols(failed_symbols_per_snapshot)
                logger.warning(summary)

            run_mode = "resume"
        elif resume:
            logger.info("[Replay] No checkpoint detected; starting fresh replay run")

        if run_mode == "fresh":
            # 防守性检查：无 checkpoint 但已有同一策略的输出文件时，拒绝静默覆盖
            replay_dir = os.path.join(VALIDATION_DIR, "replay")
            if os.path.exists(replay_dir):
                import glob
                pattern = os.path.join(
                    replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_????-??-??*.csv"
                )
                existing_files = glob.glob(pattern)
                if existing_files:
                    raise RuntimeError(
                        f"No replay checkpoint found for strategy '{strategy_slug}', "
                        f"but existing per-snapshot replay files for this strategy are already present.  "
                        "The checkpoint is the sole source of truth "
                        "for completed snapshots; per-snapshot CSV files are result carriers only "
                        "and cannot be used to infer progress or overwrite intent.  "
                        "Refusing to proceed to avoid inconsistent state.  "
                        "Please manually remove or relocate the existing output file(s) "
                        "before running the replay again."
                    )
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
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        full_df_cache: Dict[str, pd.DataFrame | None] = {}
        weekly_full_cache: Dict[str, pd.DataFrame | None] = {}
        use_weekly_cache = strategy.name in _WEEKLY_REPLAY_STRATEGIES
        SLOW_SYMBOL_THRESHOLD = 3.0

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

            # 输出该 snapshot 对应自然周的交易日历元数据（用于语义分析）
            try:
                week_info = get_snapshot_trading_week_info(snapshot_date)
                logger.info(
                    "[Replay] Snapshot %d/%d trading-week info: date=%s, "
                    "has_trading_day=%s, is_valid=%s, calendar_ok=%s, trading_days=%d",
                    snapshot_idx, len(weekly_dates), snapshot_date_str,
                    week_info["has_trading_day"],
                    week_info["is_valid_completed_trading_week"],
                    week_info["calendar_query_ok"],
                    week_info["trading_days_count"],
                )
            except Exception as exc:
                logger.debug(
                    "[Replay] Snapshot %d/%d trading-week info query failed: date=%s error=%s",
                    snapshot_idx, len(weekly_dates), snapshot_date_str, exc,
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

                snapshot_elapsed = time.time() - snapshot_start
                logger.info(
                    "[Replay] Snapshot %d/%d done (no-op): date=%s elapsed=%.1fs "
                    "processed=0 matched=0 failed=0",
                    snapshot_idx, len(weekly_dates), snapshot_date_str, snapshot_elapsed,
                )

                write_replay_results(
                    [], strategy_slug, universe, lookback_weeks, version, snapshot_date_str
                )
                write_replay_errors(
                    [], strategy_slug, universe, lookback_weeks, version, snapshot_date_str
                )

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
                    if symbol not in full_df_cache:
                        t0 = time.time()
                        current_stage = "load_full_df"
                        full_df_cache[symbol] = load_full_df_for_replay(
                            symbol, lookback_days, initial_days, request_interval, required_history_days,
                            replay_data_end_date=replay_data_end_date,
                        )
                        stage_times["load_full_df"] = time.time() - t0

                        if use_weekly_cache and full_df_cache[symbol] is not None:
                            weekly_full_cache[symbol] = precompute_weekly_bars_for_replay(
                                full_df_cache[symbol]
                            )

                    full_df = full_df_cache[symbol]

                    if full_df is None or full_df.empty:
                        exit_stage = "full_df_empty"

                    if not exit_stage:
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

                        t0 = time.time()
                        current_stage = "scan"
                        if use_weekly_cache and symbol in weekly_full_cache and weekly_full_cache[symbol] is not None:
                            # 主路径：使用预计算 weekly cache
                            precomputed_weekly = slice_weekly_bars_for_snapshot(
                                weekly_full_cache[symbol], snapshot_date,
                                snapshot_week_last_trading_day,
                            )
                            result = strategy.scan(symbol, df, context, precomputed_weekly=precomputed_weekly)
                        elif use_weekly_cache:
                            # Fallback：weekly cache 不可用时从日线实时计算（与主路径保持相同守卫逻辑）
                            from data_utils import convert_daily_to_weekly
                            weekly_full = convert_daily_to_weekly(df, cutoff_date=None)
                            precomputed_weekly = slice_weekly_bars_for_snapshot(
                                weekly_full, snapshot_date, snapshot_week_last_trading_day
                            )
                            result = strategy.scan(symbol, df, context, precomputed_weekly=precomputed_weekly)
                        else:
                            result = strategy.scan(symbol, df, context)
                        stage_times["scan"] = time.time() - t0
                        matched = result.matched

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

                    logger.error(
                        "[Replay] Snapshot %d/%d date=%s symbol %s failed at stage %s: %s: %s",
                        snapshot_idx, len(weekly_dates), snapshot_date_str, symbol, current_stage,
                        exc.__class__.__name__, str(exc)
                    )

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

                # 统一符号级可观察性日志收口
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

            # 先写文件再更新 checkpoint（崩溃时 checkpoint 不标记为已完成）
            write_replay_results(
                snapshot_results, strategy_slug, universe, lookback_weeks, version, snapshot_date_str
            )
            write_replay_errors(
                snapshot_errors, strategy_slug, universe, lookback_weeks, version, snapshot_date_str
            )

            completed_snapshot_set.add(snapshot_date_str)
            completed_snapshots.append(snapshot_date_str)
            completed_snapshots.sort()

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

        if failed_symbols_per_snapshot:
            summary = _summarize_failed_symbols(failed_symbols_per_snapshot)
            logger.warning(summary)

        logger.info("Weekly replay validation completed.")

    finally:
        bs.logout()
        logger.info("Weekly replay validation finished")


def merge_replay_snapshot_files(
    strategy_slug: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    output_path: str | None = None,
) -> str:
    """合并 per-snapshot 结果文件为单个 CSV。"""
    import glob

    replay_dir = os.path.join(VALIDATION_DIR, "replay")
    pattern = os.path.join(
        replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_????-??-??.csv"
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
            replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_merged.csv"
        )

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Merged result written to %s (%d rows)", output_path, len(merged))

    return output_path


def merge_replay_error_snapshot_files(
    strategy_slug: str,
    universe: str,
    lookback_weeks: int,
    version: str,
    output_path: str | None = None,
) -> str:
    """合并 per-snapshot 错误文件为单个 CSV。"""
    import glob

    replay_dir = os.path.join(VALIDATION_DIR, "replay")
    pattern = os.path.join(
        replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_????-??-??_errors.csv"
    )

    files = sorted(glob.glob(pattern))
    if not files:
        logger.info("No per-snapshot error files found matching: %s", pattern)
        output_path = output_path or os.path.join(
            replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_errors_merged.csv"
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
            replay_dir, f"replay_{strategy_slug}_{universe}_{lookback_weeks}w_{version}_errors_merged.csv"
        )

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Merged errors written to %s (%d rows)", output_path, len(merged))

    return output_path


if __name__ == "__main__":
    main()
