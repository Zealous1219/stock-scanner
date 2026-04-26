"""Tests for data_utils.py focusing on weekly completion logic."""

from datetime import datetime
from unittest.mock import Mock, patch
import pytest
import pandas as pd

from data_utils import (
    get_week_monday_friday,
    get_last_trading_day_of_week,
    get_last_completed_week_end,
)


class TestGetWeekMondayFriday:
    """Test get_week_monday_friday function."""

    def test_wednesday_input(self):
        """周三输入时，周一/周五计算正确"""
        # 2026-04-22 是周三
        target_date = pd.Timestamp("2026-04-22")
        monday, friday = get_week_monday_friday(target_date)

        # 本周一是 2026-04-20
        assert monday == pd.Timestamp("2026-04-20")
        # 本周五是 2026-04-24
        assert friday == pd.Timestamp("2026-04-24")

    def test_saturday_input(self):
        """周六输入时，不会漂到下周五"""
        # 2026-04-25 是周六
        target_date = pd.Timestamp("2026-04-25")
        monday, friday = get_week_monday_friday(target_date)

        # 本周一是 2026-04-20
        assert monday == pd.Timestamp("2026-04-20")
        # 本周五是 2026-04-24（不会漂到下周五）
        assert friday == pd.Timestamp("2026-04-24")


class TestGetLastTradingDayOfWeek:
    """Test get_last_trading_day_of_week function."""

    @patch("data_utils.bs.query_trade_dates")
    def test_normal_trading_week_returns_friday(self, mock_query):
        """正常交易周返回周五"""
        # Mock baostock response
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-20", "1"),  # Monday
            ("2026-04-21", "1"),  # Tuesday
            ("2026-04-22", "1"),  # Wednesday
            ("2026-04-23", "1"),  # Thursday
            ("2026-04-24", "1"),  # Friday
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        target_date = pd.Timestamp("2026-04-22")  # Wednesday
        result = get_last_trading_day_of_week(target_date)

        assert result == pd.Timestamp("2026-04-24")  # Friday

    @patch("data_utils.bs.query_trade_dates")
    def test_short_week_returns_last_trading_day(self, mock_query):
        """短周返回本周最后交易日（如周四）"""
        # Mock baostock response for holiday-shortened week
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-20", "1"),  # Monday
            ("2026-04-21", "1"),  # Tuesday
            ("2026-04-22", "1"),  # Wednesday
            ("2026-04-23", "1"),  # Thursday (Friday is holiday)
            ("2026-04-24", "0"),  # Friday (not trading)
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        target_date = pd.Timestamp("2026-04-22")  # Wednesday
        result = get_last_trading_day_of_week(target_date)

        assert result == pd.Timestamp("2026-04-23")  # Thursday

    @patch("data_utils.bs.query_trade_dates")
    def test_no_trading_days_in_week_returns_none(self, mock_query):
        """本周没有交易日时返回 None"""
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-20", "0"),  # Monday (holiday)
            ("2026-04-21", "0"),  # Tuesday (holiday)
            ("2026-04-22", "0"),  # Wednesday (holiday)
            ("2026-04-23", "0"),  # Thursday (holiday)
            ("2026-04-24", "0"),  # Friday (holiday)
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        target_date = pd.Timestamp("2026-04-22")
        result = get_last_trading_day_of_week(target_date)

        assert result is None

    @patch("data_utils.bs.query_trade_dates")
    def test_query_failure_returns_none(self, mock_query):
        """查询失败时返回 None"""
        mock_rs = Mock()
        mock_rs.error_code = "1"  # Error
        mock_rs.error_msg = "Network error"
        mock_query.return_value = mock_rs

        target_date = pd.Timestamp("2026-04-22")
        result = get_last_trading_day_of_week(target_date)

        assert result is None

    @patch("data_utils.bs.query_trade_dates")
    def test_exception_returns_none(self, mock_query):
        """baostock调用异常时返回 None"""
        mock_query.side_effect = Exception("Connection error")

        target_date = pd.Timestamp("2026-04-22")
        result = get_last_trading_day_of_week(target_date)

        assert result is None


class TestGetLastCompletedWeekEnd:
    """Test get_last_completed_week_end function."""

    def test_monday_to_thursday_returns_previous_friday(self):
        """周一到周四返回上周五"""
        # 2026-04-22 是周三
        now = datetime(2026, 4, 22, 14, 30)  # Wednesday 14:30
        daily_latest_date = pd.Timestamp("2026-04-22")

        result = get_last_completed_week_end(now, daily_latest_date)

        # 本周五: 2026-04-24, 上周五: 2026-04-17
        assert result == pd.Timestamp("2026-04-17")

    def test_friday_before_2000_returns_previous_friday(self):
        """周五 19:59 + latest_date=周五 -> 返回上周五"""
        now = datetime(2026, 4, 24, 19, 59)  # Friday 19:59
        daily_latest_date = pd.Timestamp("2026-04-24")  # Friday

        result = get_last_completed_week_end(now, daily_latest_date)

        # 本周五: 2026-04-24, 上周五: 2026-04-17
        assert result == pd.Timestamp("2026-04-17")

    def test_friday_after_2001_with_latest_date_thursday_returns_this_friday(self):
        """周五 20:01 + latest_date=周四 -> 返回本周五锚点（数据已到达本周最后交易日）"""
        now = datetime(2026, 4, 24, 20, 1)  # Friday 20:01

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            # 本周最后交易日是周四 (holiday-shortened week)
            mock_get_last.return_value = pd.Timestamp("2026-04-23")

            daily_latest_date = pd.Timestamp("2026-04-23")  # Thursday
            result = get_last_completed_week_end(now, daily_latest_date)

            # 数据已到达本周最后交易日 (周四)，返回本周五锚点
            assert result == pd.Timestamp("2026-04-24")

    def test_friday_after_2001_with_latest_date_friday_returns_this_friday(self):
        """周五 20:01 + latest_date=周五 -> 返回本周五"""
        now = datetime(2026, 4, 24, 20, 1)  # Friday 20:01

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            # 本周最后交易日是周五
            mock_get_last.return_value = pd.Timestamp("2026-04-24")

            daily_latest_date = pd.Timestamp("2026-04-24")  # Friday
            result = get_last_completed_week_end(now, daily_latest_date)

            # 数据已到达本周最后交易日 (周五)，返回本周五
            assert result == pd.Timestamp("2026-04-24")

    def test_saturday_with_normal_week_and_latest_date_thursday_returns_previous_friday(self):
        """周六（正常交易周）+ latest_date=周四 -> 返回上周五"""
        now = datetime(2026, 4, 25, 10, 0)  # Saturday 10:00

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            # 本周最后交易日是周五
            mock_get_last.return_value = pd.Timestamp("2026-04-24")

            daily_latest_date = pd.Timestamp("2026-04-23")  # Thursday
            result = get_last_completed_week_end(now, daily_latest_date)

            # 数据未到达本周最后交易日 (周五)，返回上周五
            assert result == pd.Timestamp("2026-04-17")

    def test_saturday_with_short_week_and_latest_date_last_trading_day_returns_this_friday(self):
        """周六（短周）+ latest_date=短周最后交易日 -> 返回本周五锚点"""
        now = datetime(2026, 4, 25, 10, 0)  # Saturday 10:00

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            # 本周最后交易日是周四 (holiday-shortened week)
            mock_get_last.return_value = pd.Timestamp("2026-04-23")

            daily_latest_date = pd.Timestamp("2026-04-23")  # Thursday (last trading day)
            result = get_last_completed_week_end(now, daily_latest_date)

            # 数据已到达本周最后交易日 (周四)，返回本周五锚点
            assert result == pd.Timestamp("2026-04-24")

    def test_saturday_with_short_week_and_latest_date_monday_returns_previous_friday(self):
        """周六（短周）+ latest_date=周一 -> 返回上周五"""
        now = datetime(2026, 4, 25, 10, 0)  # Saturday 10:00

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            # 本周最后交易日是周四 (holiday-shortened week)
            mock_get_last.return_value = pd.Timestamp("2026-04-23")

            daily_latest_date = pd.Timestamp("2026-04-20")  # Monday
            result = get_last_completed_week_end(now, daily_latest_date)

            # 数据未到达本周最后交易日 (周四)，返回上周五
            assert result == pd.Timestamp("2026-04-17")

    def test_daily_latest_date_none_returns_previous_friday(self):
        """daily_latest_date=None -> 返回上周五"""
        now = datetime(2026, 4, 25, 10, 0)  # Saturday 10:00
        result = get_last_completed_week_end(now, None)

        assert result == pd.Timestamp("2026-04-17")

    def test_trading_calendar_query_failure_returns_previous_friday(self):
        """交易日历查询失败/未登录时 -> 保守回退上周五"""
        now = datetime(2026, 4, 25, 10, 0)  # Saturday 10:00

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            mock_get_last.return_value = None  # Query failure

            daily_latest_date = pd.Timestamp("2026-04-24")  # Friday
            result = get_last_completed_week_end(now, daily_latest_date)

            # 查询失败，保守回退到上周五
            assert result == pd.Timestamp("2026-04-17")

    def test_default_now_parameter(self):
        """测试默认 now 参数"""
        with patch("data_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 4, 22, 14, 30)  # Wednesday
            mock_datetime.side_effect = datetime

            daily_latest_date = pd.Timestamp("2026-04-22")
            result = get_last_completed_week_end(None, daily_latest_date)

            # 周三应该返回上周五
            assert result == pd.Timestamp("2026-04-17")