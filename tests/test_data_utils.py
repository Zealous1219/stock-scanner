"""Tests for data_utils.py focusing on weekly completion logic."""

from datetime import datetime
from unittest.mock import Mock, patch
import pytest
import pandas as pd

from data_utils import (
    get_week_monday_friday,
    get_last_trading_day_of_week,
    get_last_completed_week_end,
    get_snapshot_trading_week_info,
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

    def test_friday_cutoff_depends_on_hour(self):
        """证明周五的cutoff结果依赖于hour——这是replay不稳定的根源。
        hour=8 (<20): 返回上周五
        hour=23 (>=20): 数据完整则返回本周五
        """
        daily_latest_date = pd.Timestamp("2026-04-24")  # Friday

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            mock_get_last.return_value = pd.Timestamp("2026-04-24")  # last trading day = Friday

            # Morning run: hour=8 -> previous Friday
            morning_now = datetime(2026, 4, 24, 8, 0, 0)
            morning_result = get_last_completed_week_end(morning_now, daily_latest_date)
            assert morning_result == pd.Timestamp("2026-04-17"), \
                f"Expected previous Friday, got {morning_result}"

            # Evening run: hour=23 -> this Friday (data complete)
            evening_now = datetime(2026, 4, 24, 23, 59, 59)
            evening_result = get_last_completed_week_end(evening_now, daily_latest_date)
            assert evening_result == pd.Timestamp("2026-04-24"), \
                f"Expected this Friday, got {evening_result}"

            # The results differ purely because of the hour
            assert morning_result != evening_result, (
                "Historical Friday cutoff must differ by hour — "
                "this is the instability that replay anchor fix addresses"
            )

    def test_friday_fixed_time_gives_stable_cutoff(self):
        """证明固定时间(>=20)给出稳定结果，不管外界时间如何。
        这是 replay anchor 修复后的期望行为。
        """
        daily_latest_date = pd.Timestamp("2026-04-24")  # Friday

        with patch("data_utils.get_last_trading_day_of_week") as mock_get_last:
            mock_get_last.return_value = pd.Timestamp("2026-04-24")

            # All >= 20:00 should give the same result (this Friday if data complete)
            for hour in [20, 21, 23]:
                now = datetime(2026, 4, 24, hour, 0, 0)
                result = get_last_completed_week_end(now, daily_latest_date)
                assert result == pd.Timestamp("2026-04-24"), \
                    f"hour={hour}: expected this Friday, got {result}"

            # Also test with the exact normalized replay anchor (23:59:59)
            replay_now = datetime(2026, 4, 24, 23, 59, 59)
            result = get_last_completed_week_end(replay_now, daily_latest_date)
            assert result == pd.Timestamp("2026-04-24"), \
                f"replay anchor: expected this Friday, got {result}"


class TestGetSnapshotTradingWeekInfo:
    """Test get_snapshot_trading_week_info function.

    测试历史 snapshot 的 trading-week 元数据校验能力。
    这是为改进 weekly replay 历史 snapshot 语义可解释性而新增的 helper。
    """

    @patch("data_utils.bs.query_trade_dates")
    def test_historical_week_with_trading_days_is_valid(self, mock_query):
        """历史 snapshot 周内存在交易日 -> 有效交易周"""
        # Mock baostock response: 正常交易周（周一到周五都有交易）
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

        # snapshot_date 是周五锚点
        snapshot_date = pd.Timestamp("2026-04-24")
        result = get_snapshot_trading_week_info(snapshot_date)

        # 验证关键字段
        assert result["calendar_query_ok"] is True
        assert result["has_trading_day"] is True
        assert result["is_valid_completed_trading_week"] is True
        assert result["last_trading_day_of_week"] == pd.Timestamp("2026-04-24")
        assert result["trading_days_count"] == 5
        assert result["week_start_date"] == pd.Timestamp("2026-04-20")
        assert result["week_end_date"] == pd.Timestamp("2026-04-24")
        assert result["snapshot_date"] == snapshot_date
        assert result["friday_anchor"] == snapshot_date

    @patch("data_utils.bs.query_trade_dates")
    def test_historical_week_with_no_trading_days_is_invalid(self, mock_query):
        """历史 snapshot 整周无交易 -> 无效交易周"""
        # Mock baostock response: 整周无交易（如节假日）
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

        snapshot_date = pd.Timestamp("2026-04-24")
        result = get_snapshot_trading_week_info(snapshot_date)

        # 验证关键字段：整周无交易
        assert result["calendar_query_ok"] is True
        assert result["has_trading_day"] is False
        assert result["is_valid_completed_trading_week"] is False
        assert result["last_trading_day_of_week"] is None
        assert result["trading_days_count"] == 0

    @patch("data_utils.bs.query_trade_dates")
    def test_calendar_query_failure_returns_conservative_state(self, mock_query):
        """交易日历查询失败 -> 返回明确保守状态"""
        # Mock baostock response: 查询失败
        mock_rs = Mock()
        mock_rs.error_code = "1"  # Error
        mock_rs.error_msg = "Network error"
        mock_query.return_value = mock_rs

        snapshot_date = pd.Timestamp("2026-04-24")
        result = get_snapshot_trading_week_info(snapshot_date)

        # 验证关键字段：查询失败，保守处理
        assert result["calendar_query_ok"] is False
        assert result["has_trading_day"] is False
        assert result["is_valid_completed_trading_week"] is False
        assert result["last_trading_day_of_week"] is None
        assert result["trading_days_count"] == 0

    @patch("data_utils.bs.query_trade_dates")
    def test_short_trading_week(self, mock_query):
        """短周（如周四为最后交易日）的处理"""
        # Mock baostock response: 短周（周五休市）
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.get_row_data.side_effect = [
            ("2026-04-20", "1"),  # Monday
            ("2026-04-21", "1"),  # Tuesday
            ("2026-04-22", "1"),  # Wednesday
            ("2026-04-23", "1"),  # Thursday (last trading day)
            ("2026-04-24", "0"),  # Friday (holiday)
        ]
        mock_rs.next.side_effect = [True, True, True, True, True, False]
        mock_query.return_value = mock_rs

        snapshot_date = pd.Timestamp("2026-04-24")
        result = get_snapshot_trading_week_info(snapshot_date)

        # 验证关键字段：短周但有效
        assert result["calendar_query_ok"] is True
        assert result["has_trading_day"] is True
        assert result["is_valid_completed_trading_week"] is True
        assert result["last_trading_day_of_week"] == pd.Timestamp("2026-04-23")  # Thursday
        assert result["trading_days_count"] == 4

    def test_input_datetime_type(self):
        """测试输入为 datetime 类型时也能正确处理"""
        with patch("data_utils.bs.query_trade_dates") as mock_query:
            mock_rs = Mock()
            mock_rs.error_code = "0"
            mock_rs.get_row_data.side_effect = [
                ("2026-04-20", "1"),
                ("2026-04-21", "1"),
                ("2026-04-22", "1"),
                ("2026-04-23", "1"),
                ("2026-04-24", "1"),
            ]
            mock_rs.next.side_effect = [True, True, True, True, True, False]
            mock_query.return_value = mock_rs

            # 输入为 datetime 类型
            snapshot_date = datetime(2026, 4, 24)
            result = get_snapshot_trading_week_info(snapshot_date)

            assert result["calendar_query_ok"] is True
            assert result["has_trading_day"] is True
            assert result["snapshot_date"] == pd.Timestamp("2026-04-24")

    def test_pd_timestamp_input_is_normalized(self):
        """测试 pd.Timestamp 输入（带时间分量）会被归一化到日期级"""
        with patch("data_utils.bs.query_trade_dates") as mock_query:
            mock_rs = Mock()
            mock_rs.error_code = "0"
            mock_rs.get_row_data.side_effect = [
                ("2026-04-20", "1"),
                ("2026-04-21", "1"),
                ("2026-04-22", "1"),
                ("2026-04-23", "1"),
                ("2026-04-24", "1"),
            ]
            mock_rs.next.side_effect = [True, True, True, True, True, False]
            mock_query.return_value = mock_rs

            # 输入为 pd.Timestamp，带时间分量（如 replay 中的 snapshot_date）
            snapshot_date = pd.Timestamp("2026-04-24 23:59:59")
            result = get_snapshot_trading_week_info(snapshot_date)

            # 验证：snapshot_date 和 friday_anchor 应该归一化到日期级（时间部分为 00:00:00）
            expected_date = pd.Timestamp("2026-04-24")  # 日期级，无时间分量
            assert result["snapshot_date"] == expected_date
            assert result["friday_anchor"] == expected_date
            # 验证时间分量确实被去掉了
            assert result["snapshot_date"].hour == 0
            assert result["snapshot_date"].minute == 0
            assert result["snapshot_date"].second == 0
            # 其他字段语义不变
            assert result["calendar_query_ok"] is True
            assert result["has_trading_day"] is True