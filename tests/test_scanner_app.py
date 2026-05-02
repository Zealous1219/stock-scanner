"""Tests for scanner_app.py focusing on cache refresh and stock pool fallback logic."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import os
import pytest
import pandas as pd

from scanner_app import (
    calculate_forward_returns,
    get_replay_strategy_slug,
    should_force_refresh_on_friday,
    load_or_update_data,
    get_latest_trading_day,
    get_stock_list,
)
from strategy_runtime import StrategyResult


class TestShouldForceRefreshOnFriday:
    """Test should_force_refresh_on_friday function."""

    def test_friday_before_2000_returns_false(self):
        """周五 20:00 前返回 False"""
        now = datetime(2026, 4, 24, 19, 59)  # Friday 19:59
        result = should_force_refresh_on_friday(now)
        assert result is False

    def test_friday_after_2000_returns_true(self):
        """周五 20:00 后返回 True"""
        now = datetime(2026, 4, 24, 20, 0)  # Friday 20:00
        result = should_force_refresh_on_friday(now)
        assert result is True

    def test_non_friday_returns_false(self):
        """非周五返回 False"""
        # Wednesday
        now = datetime(2026, 4, 22, 20, 0)  # Wednesday 20:00
        result = should_force_refresh_on_friday(now)
        assert result is False


class TestLoadOrUpdateData:
    """Test load_or_update_data function."""

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_friday_2030_with_latest_date_today_uses_cache(
        self, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """周五 20:30，latest_date 已是今天 -> 不调用 fetch_historical_data，直接用缓存"""
        # Mock datetime.now() to return Friday 20:30
        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 24, 20, 30)
            mock_datetime.side_effect = datetime

            # Mock file exists
            mock_exists.return_value = True

            # Mock cached data with today's date and enough rows (>60)
            # 31 (Mar) + 29 (Apr) = 60 days
            cached_df = pd.DataFrame({
                "date": [f"2026-03-{i:02d}" for i in range(1, 32)] + [f"2026-04-{i:02d}" for i in range(1, 30)],  # 60 days of data
                "open": [10.0 + i * 0.01 for i in range(60)],
                "high": [10.5 + i * 0.01 for i in range(60)],
                "low": [9.5 + i * 0.01 for i in range(60)],
                "close": [10.0 + i * 0.01 for i in range(60)],
                "volume": [1000 + i * 10 for i in range(60)],
            })
            mock_read_csv.return_value = cached_df

            # Mock ensure_daily_frame to properly convert date column
            def ensure_side_effect(df):
                print(f"ensure_daily_frame called with df['date'].dtype: {df['date'].dtype}")
                result = df.copy()
                if "date" in result.columns:
                    # Convert to Timestamp, not just datetime64
                    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
                    result = result.sort_values("date").reset_index(drop=True)
                print(f"ensure_daily_frame returning df['date'].dtype: {result['date'].dtype}")
                return result
            mock_ensure.side_effect = ensure_side_effect

            result_df, fetched = load_or_update_data(
                symbol="sh.000001",
                lookback_days=5,
                initial_days=400,
                request_interval=0.5,
            )

            # Should not call fetch_historical_data
            mock_fetch.assert_not_called()
            # Should use cache
            assert fetched is False

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_friday_2030_with_latest_date_yesterday_fetches_new_data(
        self, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """周五 20:30，latest_date 小于今天 -> 会调用 fetch_historical_data"""
        # Mock datetime.now() to return Friday 20:30
        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 24, 20, 30)
            mock_datetime.side_effect = datetime

            # Mock file exists
            mock_exists.return_value = True

            # Mock cached data with yesterday's date and enough rows (>60)
            # 31 (Mar) + 23 (Apr up to 23rd) = 54 days
            cached_df = pd.DataFrame({
                "date": [f"2026-03-{i:02d}" for i in range(1, 32)] + [f"2026-04-{i:02d}" for i in range(1, 24)],  # 54 days of data, up to 2026-04-23
                "open": [10.0 + i * 0.01 for i in range(54)],
                "high": [10.5 + i * 0.01 for i in range(54)],
                "low": [9.5 + i * 0.01 for i in range(54)],
                "close": [10.0 + i * 0.01 for i in range(54)],
                "volume": [1000 + i * 10 for i in range(54)],
            })
            mock_read_csv.return_value = cached_df

            # Mock ensure_daily_frame to properly convert date column
            def ensure_side_effect(df):
                print(f"ensure_daily_frame called with df['date'].dtype: {df['date'].dtype}")
                result = df.copy()
                if "date" in result.columns:
                    # Convert to Timestamp, not just datetime64
                    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
                    result = result.sort_values("date").reset_index(drop=True)
                print(f"ensure_daily_frame returning df['date'].dtype: {result['date'].dtype}")
                return result
            mock_ensure.side_effect = ensure_side_effect

            # Mock fetch_historical_data to return new data (already ensured)
            new_data = pd.DataFrame({
                "date": ["2026-04-24"],
                "open": [10.4],
                "high": [10.9],
                "low": [9.9],
                "close": [10.4],
                "volume": [1400],
            })
            # Mock ensure_daily_frame to handle both calls
            def ensure_side_effect(df):
                result = df.copy()
                if "date" in result.columns:
                    result["date"] = pd.to_datetime(result["date"])
                return result
            mock_ensure.side_effect = ensure_side_effect
            mock_fetch.return_value = new_data

            result_df, fetched = load_or_update_data(
                symbol="sh.000001",
                lookback_days=5,
                initial_days=400,
                request_interval=0.5,
            )

            # Should call fetch_historical_data
            mock_fetch.assert_called_once()
            # Should have performed remote fetch
            assert fetched is True
            # Check that fetch_historical_data was called with string dates
            call_args = mock_fetch.call_args
            # fetch_historical_data(symbol, start_date, end_date) uses positional args
            assert isinstance(call_args[0][1], str)  # start_date
            assert isinstance(call_args[0][2], str)  # end_date

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_fetch_historical_data_called_with_string_dates(
        self, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """传给 fetch_historical_data 的 start_date/end_date 是字符串，不是 Timestamp"""
        # Mock datetime.now() to return Wednesday (normal cache logic)
        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 22, 14, 30)  # Wednesday
            mock_datetime.side_effect = datetime

            # Mock file exists
            mock_exists.return_value = True

            # Mock cached data with old date but enough rows (>60)
            # 28 (Feb) + 31 (Mar) + 15 (Apr) = 74 days
            # Ensure date column contains string dates
            date_strings = [f"2026-02-{i:02d}" for i in range(1, 29)] + [f"2026-03-{i:02d}" for i in range(1, 32)] + [f"2026-04-{i:02d}" for i in range(1, 16)]
            cached_df = pd.DataFrame({
                "date": date_strings,  # 74 days of data, latest is 2026-04-15
                "open": [10.0 + i * 0.01 for i in range(74)],
                "high": [10.5 + i * 0.01 for i in range(74)],
                "low": [9.5 + i * 0.01 for i in range(74)],
                "close": [10.0 + i * 0.01 for i in range(74)],
                "volume": [1000 + i * 10 for i in range(74)],
            })
            mock_read_csv.return_value = cached_df

            # Mock ensure_daily_frame to properly convert date column
            def ensure_side_effect(df):
                print(f"ensure_daily_frame called with df['date'].dtype: {df['date'].dtype}")
                result = df.copy()
                if "date" in result.columns:
                    # Convert to Timestamp, not just datetime64
                    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
                    result = result.sort_values("date").reset_index(drop=True)
                print(f"ensure_daily_frame returning df['date'].dtype: {result['date'].dtype}")
                return result
            mock_ensure.side_effect = ensure_side_effect

            # Mock fetch_historical_data
            new_data = pd.DataFrame({
                "date": ["2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22"],
                "open": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                "high": [10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2],
                "low": [9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2],
                "close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                "volume": [1100, 1200, 1300, 1400, 1500, 1600, 1700],
            })
            mock_fetch.return_value = new_data

            result_df, fetched = load_or_update_data(
                symbol="sh.000001",
                lookback_days=5,
                initial_days=400,
                request_interval=0.5,
            )

            # Check that fetch_historical_data was called with string dates
            call_args = mock_fetch.call_args
            # fetch_historical_data(symbol, start_date, end_date) uses positional args
            assert isinstance(call_args[0][1], str)  # start_date
            assert isinstance(call_args[0][2], str)  # end_date
            # Note: Due to exception handling, start_date may be from initial_days (2025-03-18)
            # rather than latest_date (2026-04-15)
            # end_date should be today's date as string
            assert call_args[0][2] == "2026-04-22"

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_file_not_exists_fetches_new_data(self, mock_ensure, mock_fetch, mock_exists):
        """文件不存在时获取新数据"""
        # Mock datetime.now()
        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 22, 14, 30)
            mock_datetime.side_effect = datetime

            # Mock file doesn't exist
            mock_exists.return_value = False

            # Mock fetch_historical_data
            new_data = pd.DataFrame({
                "date": ["2026-04-20", "2026-04-21", "2026-04-22"],
                "open": [10.0, 10.1, 10.2],
                "high": [10.5, 10.6, 10.7],
                "low": [9.5, 9.6, 9.7],
                "close": [10.0, 10.1, 10.2],
                "volume": [1000, 1100, 1200],
            })
            mock_fetch.return_value = new_data

            # Mock ensure_daily_frame to return the same dataframe
            mock_ensure.return_value = new_data.copy()

            result_df, fetched = load_or_update_data(
                symbol="sh.000001",
                lookback_days=3,
                initial_days=400,
                request_interval=0.5,
            )

            # Should call fetch_historical_data
            mock_fetch.assert_called_once()
            # Should have performed remote fetch
            assert fetched is True
            # Should return tail of fetched data
            assert len(result_df) == 3

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_datetime_now_called_once_file_not_exists(
        self, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """验证文件不存在路径下 datetime.now() 只被调用一次"""
        call_counter = {"count": 0}
        fixed_now = datetime(2026, 4, 22, 14, 30)

        def counting_now():
            call_counter["count"] += 1
            return fixed_now

        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now = counting_now
            mock_datetime.side_effect = datetime
            mock_datetime.now = counting_now

            mock_exists.return_value = False
            mock_fetch.return_value = pd.DataFrame({
                "date": ["2026-04-20", "2026-04-21", "2026-04-22"],
                "open": [10.0, 10.1, 10.2],
                "high": [10.5, 10.6, 10.7],
                "low": [9.5, 9.6, 9.7],
                "close": [10.0, 10.1, 10.2],
                "volume": [1000, 1100, 1200],
            })
            mock_ensure.return_value = mock_fetch.return_value.copy()

            load_or_update_data(
                symbol="sh.000001",
                lookback_days=3,
                initial_days=400,
                request_interval=0,
            )

            assert call_counter["count"] == 1

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    @patch("scanner_app.should_force_refresh_on_friday")
    def test_datetime_now_called_once_cache_hit_path(
        self, mock_force, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """验证缓存命中路径下 datetime.now() 只被调用一次"""
        call_counter = {"count": 0}
        fixed_now = datetime(2026, 4, 24, 20, 30)  # Friday after 20:00

        def counting_now():
            call_counter["count"] += 1
            return fixed_now

        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now = counting_now
            mock_datetime.side_effect = datetime

            mock_exists.return_value = True
            cached_df = pd.DataFrame({
                "date": [f"2026-03-{i:02d}" for i in range(1, 32)] + [f"2026-04-{i:02d}" for i in range(1, 30)],
                "open": [10.0 + i * 0.01 for i in range(60)],
                "high": [10.5 + i * 0.01 for i in range(60)],
                "low": [9.5 + i * 0.01 for i in range(60)],
                "close": [10.0 + i * 0.01 for i in range(60)],
                "volume": [1000 + i * 10 for i in range(60)],
            })
            mock_read_csv.return_value = cached_df

            def ensure_side_effect(df):
                result = df.copy()
                result["date"] = pd.to_datetime(result["date"])
                return result
            mock_ensure.side_effect = ensure_side_effect
            mock_force.return_value = True

            load_or_update_data(
                symbol="sh.000001",
                lookback_days=5,
                initial_days=400,
                request_interval=0,
            )

            assert call_counter["count"] == 1

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    def test_start_date_and_today_derived_from_same_now(
        self, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """验证 start_date 和 end_date 基于同一个 now 推导"""
        fixed_now = datetime(2026, 4, 22, 14, 30)
        initial_days = 400

        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            mock_datetime.side_effect = datetime

            mock_exists.return_value = False
            mock_fetch.return_value = pd.DataFrame({
                "date": ["2026-04-22"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.0],
                "volume": [1000],
            })
            mock_ensure.return_value = mock_fetch.return_value.copy()

            load_or_update_data(
                symbol="sh.000001",
                lookback_days=1,
                initial_days=initial_days,
                request_interval=0,
            )

            call_args = mock_fetch.call_args
            start_date = call_args[0][1]
            end_date = call_args[0][2]

            expected_start = (fixed_now - timedelta(days=initial_days)).strftime("%Y-%m-%d")
            expected_end = fixed_now.strftime("%Y-%m-%d")

            assert start_date == expected_start
            assert end_date == expected_end

    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.pd.read_csv")
    @patch("scanner_app.fetch_historical_data")
    @patch("scanner_app.ensure_daily_frame")
    @patch("scanner_app.should_force_refresh_on_friday")
    def test_friday_refresh_uses_captured_now(
        self, mock_force, mock_ensure, mock_fetch, mock_read_csv, mock_exists
    ):
        """验证 should_force_refresh_on_friday 使用的是入口捕获的 now"""
        captured_args = []
        original_force = should_force_refresh_on_friday

        def capturing_force(now):
            captured_args.append(now)
            return original_force(now)

        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 24, 20, 30)
            mock_datetime.side_effect = datetime

            mock_exists.return_value = True
            cached_df = pd.DataFrame({
                "date": [f"2026-03-{i:02d}" for i in range(1, 32)] + [f"2026-04-{i:02d}" for i in range(1, 30)],
                "open": [10.0 + i * 0.01 for i in range(60)],
                "high": [10.5 + i * 0.01 for i in range(60)],
                "low": [9.5 + i * 0.01 for i in range(60)],
                "close": [10.0 + i * 0.01 for i in range(60)],
                "volume": [1000 + i * 10 for i in range(60)],
            })
            mock_read_csv.return_value = cached_df

            def ensure_side_effect(df):
                result = df.copy()
                result["date"] = pd.to_datetime(result["date"])
                return result
            mock_ensure.side_effect = ensure_side_effect
            mock_force.side_effect = capturing_force

            load_or_update_data(
                symbol="sh.000001",
                lookback_days=5,
                initial_days=400,
                request_interval=0,
            )

            assert len(captured_args) == 1
            assert captured_args[0] == datetime(2026, 4, 24, 20, 30)


class TestGetLatestTradingDay:
    """Test get_latest_trading_day function."""

    @patch("scanner_app.bs.query_trade_dates")
    def test_returns_latest_trading_day(self, mock_query):
        """get_latest_trading_day 能返回最近交易日"""
        # Mock baostock response
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-18", "1"),  # Friday (holiday)
            ("2026-04-19", "0"),  # Saturday (not trading)
            ("2026-04-20", "0"),  # Sunday (not trading)
            ("2026-04-21", "1"),  # Monday
            ("2026-04-22", "1"),  # Tuesday
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        reference_time = datetime(2026, 4, 22, 14, 30)  # Tuesday
        result = get_latest_trading_day(reference_time, lookback_days=5)

        # Should return Tuesday (2026-04-22) as it's the latest trading day
        assert result == "2026-04-22"

    @patch("scanner_app.bs.query_trade_dates")
    def test_query_failure_raises_exception(self, mock_query):
        """查询失败时抛出异常"""
        mock_rs = Mock()
        mock_rs.error_code = "1"
        mock_rs.error_msg = "Network error"
        mock_query.return_value = mock_rs

        reference_time = datetime(2026, 4, 22, 14, 30)

        with pytest.raises(RuntimeError, match="Failed to query trade dates"):
            get_latest_trading_day(reference_time)

    @patch("scanner_app.bs.query_trade_dates")
    def test_no_trading_days_raises_exception(self, mock_query):
        """没有交易日时抛出异常"""
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-18", "0"),  # Holiday
            ("2026-04-19", "0"),  # Holiday
            ("2026-04-20", "0"),  # Holiday
            ("2026-04-21", "0"),  # Holiday
            ("2026-04-22", "0"),  # Holiday
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        reference_time = datetime(2026, 4, 22, 14, 30)

        with pytest.raises(RuntimeError, match="No trading day found"):
            get_latest_trading_day(reference_time)

    def test_default_reference_time(self):
        """测试默认 reference_time 参数"""
        with patch("scanner_app.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 22, 14, 30)
            mock_datetime.side_effect = datetime

            with patch("scanner_app.bs.query_trade_dates") as mock_query:
                mock_rs = Mock()
                mock_rs.error_code = "0"
                mock_rs.get_row_data.side_effect = [("2026-04-22", "1")]
                mock_rs.next.side_effect = [True, False]
                mock_query.return_value = mock_rs

                result = get_latest_trading_day(None)

                assert result == "2026-04-22"


class TestGetStockList:
    """Test get_stock_list function."""

    @patch("scanner_app.get_latest_trading_day")
    @patch("scanner_app.bs.query_all_stock")
    def test_all_pool_uses_latest_trading_day(self, mock_query_all, mock_get_latest):
        """get_stock_list("all") 会使用 get_latest_trading_day 的结果去调用 query_all_stock"""
        # Mock get_latest_trading_day
        mock_get_latest.return_value = "2026-04-22"

        # Mock baostock response
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.fields = ["code", "code_name"]
        mock_rs.get_row_data.side_effect = [
            ("sh.000001", "平安银行"),
            ("sh.000002", "万科A"),
        ]
        mock_rs.next.side_effect = [True, True, False]
        mock_query_all.return_value = mock_rs

        result = get_stock_list("all")

        # Should call get_latest_trading_day
        mock_get_latest.assert_called_once()
        # Should call query_all_stock with the trading day
        mock_query_all.assert_called_once_with(day="2026-04-22")
        # Should return stock codes
        assert result == ["sh.000001", "sh.000002"]

    @patch("scanner_app.bs.query_hs300_stocks")
    @patch("scanner_app.logger")
    def test_hs300_pool_does_not_use_trading_day(self, mock_logger, mock_query_hs300):
        """hs300 股票池不使用 trading day"""
        # Mock baostock response
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.fields = ["code", "code_name"]
        mock_rs.get_row_data.side_effect = [
            ("sh.000001", "平安银行"),
            ("sh.000002", "万科A"),
        ]
        mock_rs.next.side_effect = [True, True, False]
        mock_query_hs300.return_value = mock_rs

        # Mock logger to avoid error logging
        mock_logger.info = Mock()
        mock_logger.error = Mock()

        # This should raise RuntimeError due to stock count validation
        with pytest.raises(RuntimeError, match="baostock returned unexpected stock count"):
            get_stock_list("hs300")

        # Should call query_hs300_stocks without day parameter
        mock_query_hs300.assert_called_once_with()

    @patch("scanner_app.bs.query_all_stock")
    def test_empty_stock_list_raises_exception(self, mock_query_all):
        """空股票列表抛出异常"""
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.fields = ["code", "code_name"]
        mock_rs.next.side_effect = [False]  # No data
        mock_query_all.return_value = mock_rs

        with patch("scanner_app.get_latest_trading_day") as mock_get_latest:
            mock_get_latest.return_value = "2026-04-22"

            with pytest.raises(RuntimeError, match="Failed to fetch stock list"):
                get_stock_list("all")

    def test_unsupported_pool_raises_exception(self):
        """不支持的股票池抛出异常"""
        with pytest.raises(ValueError, match="Unsupported stock pool"):
            get_stock_list("unknown")
class TestCalculateForwardReturns:
    """Test forward return target-date matching."""

    @staticmethod
    def _make_result(signal_date: str = "2025-01-03") -> StrategyResult:
        return StrategyResult(
            matched=True,
            reason_code="matched",
            reason_text="matched",
            details={"signal_date": signal_date},
        )

    @staticmethod
    def _make_df(rows: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [row[0] for row in rows],
                "open": [price for _, price in rows],
                "high": [price for _, price in rows],
                "low": [price for _, price in rows],
                "close": [price for _, price in rows],
                "volume": [1000] * len(rows),
            }
        )

    def test_exact_target_date_uses_exact_match(self):
        df = self._make_df(
            [
                ("2025-01-03", 100.0),
                ("2025-01-31", 110.0),
                ("2025-02-28", 120.0),
                ("2025-03-28", 130.0),
            ]
        )

        returns = calculate_forward_returns("sh.600000", self._make_result(), full_df=df)

        assert returns["return_4w"] == pytest.approx(0.10)
        assert returns["return_8w"] == pytest.approx(0.20)
        assert returns["return_12w"] == pytest.approx(0.30)

    def test_within_three_day_gap_uses_nearest_trading_day(self):
        df = self._make_df(
            [
                ("2025-01-03", 100.0),
                ("2025-01-30", 108.0),
                ("2025-03-03", 121.0),
                ("2025-03-31", 133.0),
            ]
        )

        returns = calculate_forward_returns("sh.600000", self._make_result(), full_df=df)

        assert returns["return_4w"] == pytest.approx(0.08)
        assert returns["return_8w"] == pytest.approx(0.21)
        assert returns["return_12w"] == pytest.approx(0.33)

    @patch("scanner_app.logger")
    def test_gap_beyond_three_days_keeps_nan_and_warns(self, mock_logger):
        df = self._make_df(
            [
                ("2025-01-03", 100.0),
                ("2025-02-10", 115.0),
                ("2025-03-10", 125.0),
                ("2025-04-10", 135.0),
            ]
        )

        returns = calculate_forward_returns("sh.600000", self._make_result(), full_df=df)

        assert pd.isna(returns["return_4w"])
        assert pd.isna(returns["return_8w"])
        assert pd.isna(returns["return_12w"])
        assert pd.isna(returns["return_16w"])
        assert pd.isna(returns["return_20w"])
        assert mock_logger.warning.call_count == 5

    def test_black_horse_result_with_signal_date_produces_non_nan_returns(self):
        """BlackHorse result details containing signal_date → forward returns are not all NaN."""
        signal_date = "2024-03-22"
        details = {
            "signal_type": "black_horse_ready",
            "signal_date": signal_date,
            "latest_week_end": signal_date,
            "week_3_end": signal_date,
        }
        result = StrategyResult(
            matched=True,
            reason_code="matched",
            reason_text="matched",
            details=details,
        )
        df = self._make_df([
            ("2024-03-22", 100.0),
            ("2024-04-19", 110.0),
            ("2024-05-17", 120.0),
            ("2024-06-14", 130.0),
        ])
        returns = calculate_forward_returns("sh.600000", result, full_df=df)

        assert returns["return_4w"] == pytest.approx(0.10)
        assert returns["return_8w"] == pytest.approx(0.20)
        assert returns["return_12w"] == pytest.approx(0.30)
        assert not pd.isna(returns["return_4w"])
        assert not pd.isna(returns["return_8w"])
        assert not pd.isna(returns["return_12w"])

    def test_empty_dataframe_keeps_nan(self):
        empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        returns = calculate_forward_returns("sh.600000", self._make_result(), full_df=empty_df)

        assert pd.isna(returns["return_4w"])
        assert pd.isna(returns["return_8w"])
        assert pd.isna(returns["return_12w"])


class TestReplayStrategySlug:
    def test_mr13_keeps_legacy_replay_slug(self):
        assert get_replay_strategy_slug("momentum_reversal_13") == "mr13"

    def test_black_horse_uses_strategy_name_as_replay_slug(self):
        assert get_replay_strategy_slug("black_horse") == "black_horse"
