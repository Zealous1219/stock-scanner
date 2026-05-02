"""Tests for replay observability enhancements in run_weekly_replay_validation.

Covers:
- Snapshot start/done log structure
- Symbol progress logging (per-100 and matched)
- Slow-symbol threshold warning
- Stage timing instrumentation
- Snapshot time anchor semantics (Step 4)
"""

import logging
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scanner_app import run_weekly_replay_validation, generate_weekly_snapshot_dates
from strategy_runtime import StrategyDecision, StrategyResult, StrategyContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(days: int = 120) -> pd.DataFrame:
    """Build a minimal DataFrame with `days` rows."""
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * days,
            "high": [10.5] * days,
            "low": [9.5] * days,
            "close": [10.0 + i * 0.01 for i in range(days)],
            "volume": [1000] * days,
        }
    )


def _make_matched_result() -> StrategyResult:
    return StrategyResult(
        matched=True,
        reason_code="signal",
        reason_text="matched",
        details={"signal_date": "2025-06-01", "signal_type": "momentum_reversal"},
    )


def _make_unmatched_result() -> StrategyResult:
    return StrategyResult(
        matched=False,
        reason_code="no_signal",
        reason_text="not matched",
        details={},
    )


# ---------------------------------------------------------------------------
# Test: snapshot start/done logging
# ---------------------------------------------------------------------------

class TestSnapshotLogging:
    """Verify that each snapshot emits a start log with index/date and a
    done log with elapsed/process/matched counts."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_snapshot_start_done_logs(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]

        # Two snapshot dates
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Check snapshot start logs
        start_logs = [r for r in caplog.records if "Snapshot" in r.message and "started" in r.message]
        assert len(start_logs) == 2, f"Expected 2 snapshot-start logs, got {len(start_logs)}"
        for log in start_logs:
            assert "[Replay]" in log.message
            assert "date=" in log.message

        # Check snapshot done logs
        done_logs = [r for r in caplog.records if "Snapshot" in r.message and "done" in r.message]
        assert len(done_logs) == 2, f"Expected 2 snapshot-done logs, got {len(done_logs)}"
        for log in done_logs:
            assert "elapsed=" in log.message
            assert "processed=" in log.message
            assert "matched=" in log.message


# ---------------------------------------------------------------------------
# Test: symbol progress logging (per-100 and matched)
# ---------------------------------------------------------------------------

class TestSymbolProgressLogging:
    """Verify that symbol-level logs are emitted at per-100 boundaries and on match."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_matched_symbol_log(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]

        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        symbol_logs = [r for r in caplog.records if "[Replay]" in r.message and "symbol" in r.message.lower()]
        assert any("matched=True" in r.message for r in symbol_logs), \
            "Expected a symbol log with matched=True"

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns")
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_per_100_progress_log(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

        # 101 stocks to trigger the 100th progress log
        stocks = [f"sh.6000{i:03d}" for i in range(101)]
        mock_stocks.return_value = stocks

        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Find the per-100 progress log
        progress_logs = [
            r for r in caplog.records
            if "[Replay]" in r.message and "symbol 100/101" in r.message
        ]
        assert len(progress_logs) >= 1, "Expected a progress log at symbol index 100"


# ---------------------------------------------------------------------------
# Test: slow symbol warning
# ---------------------------------------------------------------------------

class TestSlowSymbolWarning:
    """Verify that a symbol whose total processing time exceeds the threshold
    triggers a WARNING-level log with required fields."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.time.time")
    def test_slow_symbol_warning_logged(
        self, mock_time, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Deterministic time sequence matching the exact call order in
        # the refactored symbol loop.  For a single matched symbol with
        # a cache-miss on full_df:
        #
        #   1: snapshot_start        -> 1000.0
        #   2: symbol_start          -> 1000.0
        #   3: t0  load_full_df      -> 1000.01
        #   4: t1  load_full_df end  -> 1000.01
        #   5: t0  load_historical   -> 1000.01
        #   6: t1  load_historical   -> 1000.01
        #   7: t0  scan              -> 1000.01
        #   8: t1  scan end          -> 1000.01
        #   9: t0  calc_returns      -> 1000.01
        #  10: t1  calc_returns end  -> 1000.01
        #  11: symbol_elapsed        -> 1005.0   (5.0s > 3.0s threshold)
        #  12: snapshot_elapsed      -> 1005.0
        time_values = iter([
            1000.0,   # 1 snapshot_start
            1000.0,   # 2 symbol_start
            1000.01,  # 3 t0 load_full_df
            1000.01,  # 4 t1 load_full_df end
            1000.01,  # 5 t0 load_historical
            1000.01,  # 6 t1 load_historical end
            1000.01,  # 7 t0 scan
            1000.01,  # 8 t1 scan end
            1000.01,  # 9 t0 calc_returns
            1000.01,  # 10 t1 calc_returns end
            1005.0,   # 11 symbol_elapsed  — 5.0s > SLOW_SYMBOL_THRESHOLD
            1005.0,   # 12 snapshot_elapsed
        ])
        mock_time.side_effect = lambda: next(time_values)

        with caplog.at_level(logging.WARNING):
            run_weekly_replay_validation()

        slow_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "Slow symbol" in r.message]
        assert len(slow_warnings) >= 1, (
            f"Expected at least 1 slow-symbol WARNING, got {len(slow_warnings)}. "
            f"All WARNING+ records: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )
        w = slow_warnings[0]
        assert "[Replay]" in w.message, f"Missing [Replay] prefix: {w.message}"
        assert "sh.600000" in w.message, f"Missing symbol name: {w.message}"
        # Must contain at least one stage timing field
        assert any(k in w.message for k in ("load_historical=", "scan=", "calc_returns=")), \
            f"Missing stage timing field in warning: {w.message}"


# ---------------------------------------------------------------------------
# Test: stage timing keys are recorded
# ---------------------------------------------------------------------------

class TestStageTiming:
    """Verify that stage-times dict is populated for the key processing stages.

    We can't easily inspect the local `stage_times` dict, so we verify indirectly
    through log messages that include stage names like `scan=` and through
    the slow-symbol warning containing stage names.
    """

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_matched_symbol_log_includes_scan_time(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        symbol_logs = [
            r for r in caplog.records
            if "[Replay]" in r.message and "symbol" in r.message and "matched=True" in r.message
        ]
        assert len(symbol_logs) >= 1, "Expected at least one matched symbol log"
        # Verify the log includes scan timing
        for log in symbol_logs:
            assert "scan=" in log.message, f"Expected scan= timing in log: {log.message}"


# ---------------------------------------------------------------------------
# Test: processed count in snapshot done log
# ---------------------------------------------------------------------------

class TestProcessedCount:
    """Verify processed_count reflects symbols that passed both data filters."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns")
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_processed_count_excludes_empty_data(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: one with None full_df, one with None df, one valid
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        full_df_valid = _make_df()
        call_count = [0]

        def mock_full_df(symbol, *args, **kwargs):
            call_count[0] += 1
            if symbol == "sh.000001":
                return None  # empty — skipped
            return full_df_valid

        mock_load_full.side_effect = mock_full_df

        def mock_hist_df(symbol, *args, **kwargs):
            if symbol == "sh.000002":
                return None  # empty after historical filter — skipped
            return full_df_valid

        mock_load_hist.side_effect = mock_hist_df

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        done_logs = [r for r in caplog.records if "[Replay]" in r.message and "done" in r.message]
        assert len(done_logs) == 1
        # Only sh.000003 was processed (passed both full_df and df filters)
        assert "processed=1" in done_logs[0].message, f"Expected processed=1, got: {done_logs[0].message}"


class TestEarlyExitObservability:
    """Verify that symbols exiting early (empty full_df or empty df) still
    receive per-100 progress logging and slow-warning coverage."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns")
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.time.time")
    @patch("scanner_app.bs")
    def test_early_exit_full_df_empty_in_per100_log(
        self, mock_bs, mock_time, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """A symbol whose full_df is None appears at per-100 boundary with exit=full_df_empty."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

        # 100 stocks, all returning None full_df
        stocks = [f"sh.{i:06d}" for i in range(100)]
        mock_stocks.return_value = stocks
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        mock_create.return_value = strategy

        mock_load_full.return_value = None  # All full_df are None

        # Stable monotonic time
        tn = [0.0]
        def tick():
            tn[0] += 0.01
            return tn[0]
        mock_time.side_effect = tick

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # The per-100 log at symbol idx=100 should exist and include exit=full_df_empty
        per100_logs = [
            r for r in caplog.records
            if "[Replay]" in r.message and "symbol 100/100" in r.message
        ]
        assert len(per100_logs) >= 1, f"Expected per-100 log at symbol 100, got: {[r.message for r in caplog.records if '[Replay]' in r.message]}"
        assert "exit=full_df_empty" in per100_logs[0].message, \
            f"Expected exit=full_df_empty in per-100 log: {per100_logs[0].message}"

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns")
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.time.time")
    @patch("scanner_app.bs")
    def test_early_exit_hist_empty_in_slow_warning(
        self, mock_bs, mock_time, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """A symbol exiting at hist_empty can still trigger a slow-warning."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

        mock_stocks.return_value = ["sh.999999"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        mock_create.return_value = strategy

        mock_load_full.return_value = _make_df()
        mock_load_hist.return_value = None  # hist_empty exit

        # Deterministic time: symbol takes >3s total
        time_values = iter([
            1000.0,   # snapshot_start
            1000.0,   # symbol_start
            1000.01,  # t0 load_full_df (cache miss)
            1000.01,  # t1 load_full_df end
            1000.01,  # t0 load_historical
            1000.01,  # t1 load_historical end
            1005.0,   # symbol_elapsed — 5.0s > 3.0 threshold
            1005.0,   # snapshot_elapsed
        ])
        mock_time.side_effect = lambda: next(time_values)

        with caplog.at_level(logging.WARNING):
            run_weekly_replay_validation()

        slow_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "Slow symbol" in r.message]
        assert len(slow_warnings) >= 1, (
            f"Expected slow-symbol WARNING for early-exit symbol. "
            f"Records: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )
        w = slow_warnings[0]
        assert "[Replay]" in w.message
        assert "sh.999999" in w.message
        assert "exit=hist_empty" in w.message, f"Expected exit=hist_empty: {w.message}"
        # Stage timings should still be present (load_full_df from cache miss)
        assert "load_full_df=" in w.message or "load_historical=" in w.message, \
            f"Expected at least one stage timing: {w.message}"

class TestFaultTolerance:
    """Verify per-symbol exception isolation in replay validation."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_load_full_df_exception_isolation(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Exception in load_full_df_for_replay should isolate to current symbol."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: one fails, two succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Make load_full_df_for_replay raise exception for sh.000002
        def mock_full_df(symbol, *args, **kwargs):
            if symbol == "sh.000002":
                raise RuntimeError("Mocked full_df load failure")
            return _make_df()

        mock_load_full.side_effect = mock_full_df
        mock_load_hist.return_value = _make_df()

        with caplog.at_level(logging.ERROR):
            run_weekly_replay_validation()

        # Verify error was logged
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and "failed at stage" in r.message]
        assert len(error_logs) == 1, f"Expected 1 error log, got {len(error_logs)}"
        assert "sh.000002" in error_logs[0].message
        assert "load_full_df" in error_logs[0].message
        assert "RuntimeError" in error_logs[0].message

        # Verify error CSV was written
        mock_write_errors.assert_called_once()
        # Verify replay results were still written (for successful symbols)
        mock_write_results.assert_called_once()

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_strategy_scan_exception_isolation(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Exception in strategy.scan should isolate to current symbol."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: one fails in scan, two succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")

        # Make scan raise exception for sh.000002
        def mock_scan(symbol, df, context, **kwargs):
            if symbol == "sh.000002":
                raise ValueError("Mocked scan failure")
            return _make_matched_result()

        strategy.scan.side_effect = mock_scan
        mock_create.return_value = strategy

        mock_load_hist.return_value = _make_df()

        with caplog.at_level(logging.ERROR):
            run_weekly_replay_validation()

        # Verify error was logged
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and "failed at stage" in r.message]
        assert len(error_logs) == 1, f"Expected 1 error log, got {len(error_logs)}"
        assert "sh.000002" in error_logs[0].message
        assert "scan" in error_logs[0].message
        assert "ValueError" in error_logs[0].message

        # Verify error CSV was written
        mock_write_errors.assert_called_once()
        # Verify replay results were still written (for successful symbols)
        mock_write_results.assert_called_once()

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns")
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_calculate_forward_returns_exception_isolation(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Exception in calculate_forward_returns should isolate to current symbol."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: one fails in calculate_forward_returns, two succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Make calculate_forward_returns raise exception for sh.000002
        def _calc_returns_side_effect(symbol, result, full_df=None):
            if symbol == "sh.000002":
                raise RuntimeError("Mocked forward returns calculation failure")
            return {"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15}

        mock_calc_returns.side_effect = _calc_returns_side_effect

        run_weekly_replay_validation()

        # Verify error CSV was written (proves exception was caught and recorded)
        assert mock_write_errors.call_count == 1, f"Expected write_replay_errors to be called once, was called {mock_write_errors.call_count} times"

        # Verify replay results were still written (for successful symbols)
        mock_write_results.assert_called_once()

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_early_exit_not_counted_as_failed(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Symbols with empty full_df or df (early exit) should not be counted as failed."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: one empty full_df, one empty df, one valid
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # sh.000001: empty full_df, sh.000002: empty df, sh.000003: valid
        def mock_full_df(symbol, *args, **kwargs):
            if symbol == "sh.000001":
                return None
            return _make_df()

        def mock_hist_df(symbol, *args, **kwargs):
            if symbol == "sh.000002":
                return None
            return _make_df()

        mock_load_full.side_effect = mock_full_df
        mock_load_hist.side_effect = mock_hist_df

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Verify snapshot done log contains failed=0
        done_logs = [r for r in caplog.records if "[Replay]" in r.message and "done" in r.message]
        assert len(done_logs) == 1
        assert "failed=0" in done_logs[0].message, f"Expected failed=0, got: {done_logs[0].message}"
        # Only sh.000003 should be processed
        assert "processed=1" in done_logs[0].message

        # Verify error handler was called (to handle stale file cleanup)
        mock_write_errors.assert_called_once()
        # Verify it was called with empty errors list
        call_args = mock_write_errors.call_args
        assert call_args is not None, "write_replay_errors should have been called"
        errors_arg = call_args[0][0]  # First positional arg is errors list
        assert errors_arg == [], f"Expected empty errors list, got: {errors_arg}"
        # Verify replay results were written for the valid symbol
        mock_write_results.assert_called_once()

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_snapshot_done_log_includes_failed_count(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Snapshot done log should include failed count when exceptions occur."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 5 stocks: 2 fail, 3 succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003", "sh.000004", "sh.000005"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")

        # Make scan raise exception for sh.000002 and sh.000004
        def mock_scan(symbol, df, context, **kwargs):
            if symbol in ["sh.000002", "sh.000004"]:
                raise ValueError(f"Mocked failure for {symbol}")
            return _make_matched_result()

        strategy.scan.side_effect = mock_scan
        mock_create.return_value = strategy

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Verify snapshot done log contains failed=2
        done_logs = [r for r in caplog.records if "[Replay]" in r.message and "done" in r.message]
        assert len(done_logs) == 1
        assert "failed=2" in done_logs[0].message, f"Expected failed=2, got: {done_logs[0].message}"
        # All 5 symbols passed full_df and df filters, so processed=5
        assert "processed=5" in done_logs[0].message

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date")
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_load_historical_exception_isolation(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, caplog,
    ):
        """Exception in load_historical_data_up_to_date isolates to current symbol;
        subsequent symbols continue and snapshot is not interrupted."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: sh.000002 fails at load_historical, the other two succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def mock_hist_df(symbol, *args, **kwargs):
            if symbol == "sh.000002":
                raise RuntimeError("Mocked load_historical failure")
            return _make_df()

        mock_load_hist.side_effect = mock_hist_df

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Error was logged with correct stage
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and "failed at stage" in r.message]
        assert len(error_logs) == 1, f"Expected 1 error log, got {len(error_logs)}"
        assert "sh.000002" in error_logs[0].message
        assert "load_historical" in error_logs[0].message
        assert "RuntimeError" in error_logs[0].message
        # snapshot_date must appear in the error log
        assert "2025-05-09" in error_logs[0].message, \
            f"Expected snapshot_date in error log: {error_logs[0].message}"

        # Error CSV written (proves exception was caught and recorded)
        mock_write_errors.assert_called_once()

        # Replay results written for the two successful symbols
        mock_write_results.assert_called_once()

        # Snapshot done log shows failed=1
        done_logs = [r for r in caplog.records if "[Replay]" in r.message and "done" in r.message]
        assert len(done_logs) == 1
        assert "failed=1" in done_logs[0].message, f"Expected failed=1, got: {done_logs[0].message}"


class TestReplayCheckpointResume:
    """Verify snapshot-level replay checkpoint/resume semantics."""

    def _base_config(self):
        return {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_first_run_creates_checkpoint_after_completed_snapshots(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, mock_save_checkpoint,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def exists_side_effect(path):
            return False

        mock_exists.side_effect = exists_side_effect

        run_weekly_replay_validation()

        assert mock_write_results.call_count == 2
        assert mock_save_checkpoint.call_count == 2

        first_call = mock_save_checkpoint.call_args_list[0].args
        second_call = mock_save_checkpoint.call_args_list[1].args

        assert first_call[1] == "mr13_all_52w_v1"
        assert first_call[2] == "all"
        assert first_call[3] == 52
        assert first_call[4] == "v1"
        assert first_call[5] == ["2025-05-02"]
        assert first_call[6] == ["2025-05-02", "2025-05-09"]
        assert second_call[5] == ["2025-05-02", "2025-05-09"]
        assert second_call[6] == ["2025-05-02", "2025-05-09"]

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_resume_skips_completed_snapshots_and_only_process_pending(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_write_errors, mock_ensure,
        mock_exists, mock_getsize, mock_save_checkpoint, mock_load_checkpoint,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9), datetime(2025, 5, 16)]
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": ["2025-05-02", "2025-05-09"],
            "replay_data_end_date": "2025-10-10",  # 2025-05-16 + 21 weeks = 2025-10-10
        }

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect

        run_weekly_replay_validation()

        assert strategy.scan.call_count == 1
        processed_snapshot_dates = [call.args[2].now.strftime("%Y-%m-%d") for call in strategy.scan.call_args_list]
        assert processed_snapshot_dates == ["2025-05-16"]
        mock_write_results.assert_called_once()
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args.args[5] == ["2025-05-02", "2025-05-09", "2025-05-16"]

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_interrupted_snapshot_is_not_marked_complete_and_runs_again_on_resume(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, mock_save_checkpoint, mock_load_checkpoint,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        checkpoint_payload = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",  # 2025-05-09 + 21 weeks (adjusted for test)
        }

        def exists_side_effect(path):
            return path.endswith(".csv") or path.endswith(".checkpoint.json")

        mock_exists.side_effect = exists_side_effect

        def write_results_side_effect(results, strategy_slug, universe, lookback_weeks, version, write_header=True):
            snapshot_date = results[0]["snapshot_date"]
            if snapshot_date == "2025-05-09":
                raise RuntimeError("simulated interruption during snapshot write")
            return "replay.csv"

        mock_write_results.side_effect = write_results_side_effect

        with patch("scanner_app.load_replay_checkpoint", return_value={
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": [],
            "replay_data_end_date": "2025-10-03",  # 2025-05-09 + 21 weeks
        }):
            try:
                run_weekly_replay_validation()
            except RuntimeError as exc:
                assert "simulated interruption" in str(exc)
            else:
                raise AssertionError("Expected simulated interruption to abort the run")

        saved_snapshot_lists = [call.args[5] for call in mock_save_checkpoint.call_args_list]
        assert saved_snapshot_lists == [["2025-05-02"]]

        strategy.scan.reset_mock()
        mock_write_results.reset_mock()
        mock_save_checkpoint.reset_mock()
        mock_load_checkpoint.return_value = checkpoint_payload

        def write_results_resume(results, strategy_slug, universe, lookback_weeks, version, write_header=True):
            return "replay.csv"

        mock_write_results.side_effect = write_results_resume

        run_weekly_replay_validation()

        processed_snapshot_dates = [call.args[2].now.strftime("%Y-%m-%d") for call in strategy.scan.call_args_list]
        assert processed_snapshot_dates == ["2025-05-09"]
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args.args[5] == ["2025-05-02", "2025-05-09"]

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    def test_resume_rejects_mismatched_checkpoint_parameters(
        self, mock_bs, mock_create, mock_config, mock_stocks, mock_dates,
        mock_ensure, mock_exists, mock_load_checkpoint,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2)]
        mock_create.return_value = MagicMock()
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_26w_v1",
            "universe": "all",
            "lookback_weeks": 26,
            "version": "v1",
            "snapshot_dates": ["2025-05-02"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-10",  # Add to avoid missing schema error
        }

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect

        try:
            run_weekly_replay_validation()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected mismatched checkpoint to be rejected")

        assert "does not match current experiment parameters" in message
        assert "lookback_weeks=26" in message

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.build_replay_record")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_build_replay_record_exception_isolation(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_build_record, mock_write_errors, mock_write_results,
        mock_ensure, mock_exists, mock_getsize, caplog,
    ):
        """Exception in build_replay_record isolates to current symbol and is
        recorded with stage='build_record', not 'calc_returns'."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        # 3 stocks: sh.000002 fails at build_replay_record, the other two succeed
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def mock_build(symbol, *args, **kwargs):
            if symbol == "sh.000002":
                raise ValueError("Mocked build_replay_record failure")
            return {"symbol": symbol, "snapshot_date": "2025-05-09"}

        mock_build_record.side_effect = mock_build

        with caplog.at_level(logging.INFO):
            run_weekly_replay_validation()

        # Error was logged with stage=build_record (not calc_returns)
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and "failed at stage" in r.message]
        assert len(error_logs) == 1, f"Expected 1 error log, got {len(error_logs)}"
        assert "sh.000002" in error_logs[0].message
        assert "build_record" in error_logs[0].message, \
            f"Expected stage 'build_record' in error log, got: {error_logs[0].message}"
        assert "calc_returns" not in error_logs[0].message, \
            f"Stage must not be 'calc_returns': {error_logs[0].message}"

        # Error CSV written; verify the stage field in the error record
        mock_write_errors.assert_called_once()
        error_records = mock_write_errors.call_args[0][0]
        assert len(error_records) == 1
        assert error_records[0]["stage"] == "build_record", \
            f"Expected stage='build_record' in error record, got: {error_records[0]['stage']}"

        # Replay results written for the two successful symbols
        mock_write_results.assert_called_once()

        # Snapshot done log shows failed=1
        done_logs = [r for r in caplog.records if "[Replay]" in r.message and "done" in r.message]
        assert len(done_logs) == 1
        assert "failed=1" in done_logs[0].message, f"Expected failed=1, got: {done_logs[0].message}"

    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_no_checkpoint_with_existing_replay_output_raises_error(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results, mock_ensure,
        mock_getsize, mock_load_checkpoint, mock_save_checkpoint,
    ):
        """When resume=True and no checkpoint exists but per-snapshot result files
        already exist, a RuntimeError must be raised and scan must never execute."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Create a temporary directory with existing per-snapshot files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create an existing per-snapshot result file for mr13 strategy
            existing_file = os.path.join(
                replay_dir, "replay_mr13_all_52w_v1_2025-05-02.csv"
            )
            pd.DataFrame({"test": [1]}).to_csv(existing_file, index=False)

            # Patch VALIDATION_DIR to use temp directory
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Use real os.path.exists - no mock needed
                # The temporary directory and files are real, so os.path.exists
                # will correctly detect them.

                try:
                    run_weekly_replay_validation()
                except RuntimeError as exc:
                    message = str(exc)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    raise
                else:
                    raise AssertionError("Expected RuntimeError when per-snapshot result file exists but no checkpoint")

                assert "checkpoint" in message.lower(), f"Error message must mention checkpoint: {message}"
                assert "existing" in message.lower(), f"Error message must mention existing output: {message}"
                assert "refusing" in message.lower(), f"Error message must mention refusing: {message}"
                assert "manually" in message.lower(), f"Error message must mention manual action: {message}"
                strategy.scan.assert_not_called()

    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_can_run_false_snapshot_enters_checkpoint_and_skipped_on_resume(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, mock_load_checkpoint, mock_save_checkpoint,
    ):
        """can_run=False snapshot: scan is NOT called, snapshot enters
        checkpoint, and is skipped on resume."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [
            datetime(2025, 5, 2),
            datetime(2025, 5, 9),
            datetime(2025, 5, 16),
        ]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        # S1: True, S2: False, S3: True
        strategy.can_run.side_effect = [
            StrategyDecision(should_run=True, reason_code="ok", reason_text="ok"),
            StrategyDecision(should_run=False, reason_code="incomplete_week", reason_text="skipped"),
            StrategyDecision(should_run=True, reason_code="ok", reason_text="ok"),
        ]
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # First run: no checkpoint, no CSV
        def exists_side_effect_first(path):
            return False

        mock_exists.side_effect = exists_side_effect_first

        run_weekly_replay_validation()

        # scan was only called for S1(2025-05-02) and S3(2025-05-16), NOT S2
        scanned_dates = [call.args[2].now.strftime("%Y-%m-%d") for call in strategy.scan.call_args_list]
        assert "2025-05-02" in scanned_dates, f"Expected scan on 2025-05-02, got {scanned_dates}"
        assert "2025-05-09" not in scanned_dates, f"S2(can_run=False) must not have scan: {scanned_dates}"
        assert "2025-05-16" in scanned_dates, f"Expected scan on 2025-05-16, got {scanned_dates}"
        assert len(scanned_dates) == 2, f"Expected exactly 2 scans, got {len(scanned_dates)}: {scanned_dates}"

        # save_replay_checkpoint was called 3 times (once per snapshot including the no-op)
        assert mock_save_checkpoint.call_count == 3
        saved_lists = [call.args[5] for call in mock_save_checkpoint.call_args_list]
        assert "2025-05-09" in saved_lists[1], f"S2 must be in checkpoint: {saved_lists}"
        final_list_first_run = saved_lists[-1]
        assert final_list_first_run == ["2025-05-02", "2025-05-09", "2025-05-16"], \
            f"Expected all three snapshots, got {final_list_first_run}"

        # Setup resume: checkpoint exists with all 3 completed
        strategy.scan.reset_mock()
        mock_save_checkpoint.reset_mock()
        mock_write_results.reset_mock()

        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "replay_data_end_date": "2025-10-10",  # 2025-05-16 + 21 weeks
        }

        def exists_side_effect_resume(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect_resume

        mock_getsize.return_value = 10  # csv is non-empty

        run_weekly_replay_validation()

        # On resume, scan must not be called (all snapshots already completed)
        strategy.scan.assert_not_called()
        mock_save_checkpoint.assert_not_called()

    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_can_run_false_cleans_up_stale_error_file(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_write_errors, mock_ensure,
        mock_exists, mock_getsize, mock_load_checkpoint, mock_save_checkpoint,
    ):
        """When a snapshot that previously produced errors is re-run with
        can_run=False, the stale error file is cleaned up, the empty result
        carrier file is preserved, and the checkpoint is updated."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        # First run: both snapshots can_run=True, S2 scan produces error
        strategy.can_run.side_effect = [
            StrategyDecision(should_run=True, reason_code="ok", reason_text="ok"),
            StrategyDecision(should_run=True, reason_code="ok", reason_text="ok"),
        ]
        strategy.scan.side_effect = [
            _make_matched_result(),
            RuntimeError("simulated scan failure for S2"),
        ]
        mock_create.return_value = strategy

        mock_exists.side_effect = lambda path: False

        run_weekly_replay_validation()

        # S1: no errors → write_replay_errors([], ...)
        # S2: scan fails → write_replay_errors([{...}], ...)
        assert mock_write_errors.call_count == 2
        s1_errors = mock_write_errors.call_args_list[0][0][0]
        assert len(s1_errors) == 0
        s2_errors = mock_write_errors.call_args_list[1][0][0]
        assert len(s2_errors) == 1
        assert s2_errors[0]["code"] == "sh.600000"
        assert "simulated scan failure" in s2_errors[0]["error_message"]
        assert mock_write_errors.call_args_list[1][0][5] == "2025-05-09"

        # Both snapshots saved to checkpoint
        assert mock_save_checkpoint.call_count == 2

        # ----- Second run: S2 is pending, can_run=False -----
        strategy.scan.reset_mock()
        strategy.can_run.reset_mock()
        mock_write_results.reset_mock()
        mock_write_errors.reset_mock()
        mock_save_checkpoint.reset_mock()

        strategy.can_run.side_effect = lambda ctx: StrategyDecision(
            should_run=ctx.now.strftime("%Y-%m-%d") != "2025-05-09",
            reason_code="incomplete_week" if ctx.now.strftime("%Y-%m-%d") == "2025-05-09" else "ok",
            reason_text="skipped" if ctx.now.strftime("%Y-%m-%d") == "2025-05-09" else "ok",
        )

        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",
        }

        mock_exists.side_effect = lambda path: (
            path.endswith(".checkpoint.json") or path.endswith(".csv")
        )

        run_weekly_replay_validation()

        # S2 (can_run=False) calls write_replay_errors([], ...)
        # to clean up the stale error file from the first run
        mock_write_errors.assert_called_once()
        noop_errors = mock_write_errors.call_args[0][0]
        assert noop_errors == [], "No-op snapshot must call write_replay_errors([])"
        assert mock_write_errors.call_args[0][5] == "2025-05-09"

        # Empty result carrier file is also written
        mock_write_results.assert_called_once()
        noop_results = mock_write_results.call_args[0][0]
        assert noop_results == [], "No-op snapshot must call write_replay_results([])"
        assert mock_write_results.call_args[0][5] == "2025-05-09"

        # scan is NOT called (can_run=False)
        strategy.scan.assert_not_called()

        # Checkpoint updated to include S2
        mock_save_checkpoint.assert_called_once()
        assert mock_save_checkpoint.call_args.args[5] == ["2025-05-02", "2025-05-09"]


# ---------------------------------------------------------------------------
# Step 4: Snapshot time anchor semantic stability
# ---------------------------------------------------------------------------

class TestSnapshotTimeAnchor:
    """Verify that replay snapshot dates carry a fixed time so that
    weekly cutoff decisions are stable, and that the completed-week
    boundary excludes the unfinished current week."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_friday_mock(hour: int, minute: int = 0) -> MagicMock:
        """Build a MagicMock that behaves like datetime.datetime in
        scanner_app for a Friday at the given hour:minute."""
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 5, 1, hour, minute, 0)
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min  # so .min.time().replace(...) works
        return mock_dt

    # ------------------------------------------------------------------
    # Fixed-time anchor tests (pre-existing, adapted)
    # ------------------------------------------------------------------

    def test_generate_snapshot_dates_always_fixed_time(self):
        """Regardless of runtime hour, every returned Friday has hour=23:59:59."""
        # Friday morning — current week excluded, so last snapshot is last week
        mock_dt = self._make_friday_mock(hour=8, minute=15)

        with patch("scanner_app.datetime", mock_dt):
            dates = generate_weekly_snapshot_dates(4)

        assert len(dates) > 0, "Expected at least one Friday snapshot"
        for d in dates:
            assert d.weekday() == 4, f"Expected Friday, got weekday={d.weekday()}"
            assert d.hour == 23, f"Expected hour=23, got {d.hour} for {d}"
            assert d.minute == 59, f"Expected minute=59, got {d.minute} for {d}"
            assert d.second == 59, f"Expected second=59, got {d.second} for {d}"

    def test_same_completion_zone_gives_identical_results(self):
        """Within the same completion zone (e.g. Friday morning vs Friday
        afternoon), different hours produce the same snapshot list."""
        results = {}
        for mock_hour in [7, 14]:
            mock_dt = self._make_friday_mock(hour=mock_hour)
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)
            results[mock_hour] = dates

        expected = results[7]
        for hour, dates in results.items():
            assert len(dates) == len(expected), f"hour={hour}: length mismatch"
            for i, d in enumerate(dates):
                assert d == expected[i], (
                    f"hour={hour} snapshot {i} differs: {d} != {expected[i]}"
                )

    # ------------------------------------------------------------------
    # Completion-boundary tests (new for Step 4 boundary fix)
    # ------------------------------------------------------------------

    def test_friday_morning_excludes_current_week(self):
        """Friday before 20:00 must NOT include the current week's Friday
        as a snapshot — it is not yet completed."""
        # Friday 2026-05-01 08:00 — this week is incomplete
        mock_dt = self._make_friday_mock(hour=8)
        with patch("scanner_app.datetime", mock_dt):
            dates = generate_weekly_snapshot_dates(4)

        # The last Friday must be 2026-04-24, not 2026-05-01
        last = dates[-1]
        assert last.date().isoformat() == "2026-04-24", (
            f"Expected last snapshot 2026-04-24, got {last.date()}"
        )
        # No snapshot should be 2026-05-01
        for d in dates:
            assert d.date().isoformat() != "2026-05-01", (
                f"2026-05-01 must not appear when run on Friday morning: {d}"
            )

    def test_friday_evening_includes_current_week(self):
        """Friday after 20:00 MUST include the current week's Friday
        because the week has completed."""
        # Friday 2026-05-01 21:00 — this week is complete
        mock_dt = self._make_friday_mock(hour=21)
        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-05-01")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        # The last Friday must be 2026-05-01
        last = dates[-1]
        assert last.date().isoformat() == "2026-05-01", (
            f"Expected last snapshot 2026-05-01, got {last.date()}"
        )

    def test_monday_runtime_last_snapshot_is_last_friday(self):
        """Running on Monday must have the previous Friday as the last snapshot."""
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 5, 4, 10, 0, 0)  # Monday
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.datetime", mock_dt):
            dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        assert last.date().isoformat() == "2026-05-01", (
            f"Expected last snapshot 2026-05-01 (last Friday), got {last.date()}"
        )

    def test_sunday_runtime_last_snapshot_is_just_passed_friday(self):
        """Running on Sunday must have the just-passed Friday as the last snapshot."""
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 5, 3, 10, 0, 0)  # Sunday
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-05-01")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        assert last.date().isoformat() == "2026-05-01", (
            f"Expected last snapshot 2026-05-01 (just-passed Friday), got {last.date()}"
        )

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_replay_context_now_has_fixed_time(
        self, mock_bs, mock_config, mock_stocks,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize,
    ):
        """In replay, context.now passed to strategy.scan must carry
        the normalized time (hour=23:59:59), not the runtime time."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Mock generate_weekly_snapshot_dates to return a Friday with a
        # specific date but with random runtime time — the function itself
        # is now responsible for normalising it.
        with patch("scanner_app.generate_weekly_snapshot_dates") as mock_dates:
            # Simulate what the real function now does: Friday date + fixed time
            mock_dates.return_value = [
                datetime(2025, 5, 2, 23, 59, 59),
                datetime(2025, 5, 9, 23, 59, 59),
            ]
            run_weekly_replay_validation()

        assert strategy.scan.call_count == 2
        for call in strategy.scan.call_args_list:
            ctx = call.args[2]  # StrategyContext is the 3rd arg to scan
            assert ctx.now.hour == 23, f"Expected hour=23 in context.now, got {ctx.now}"
            assert ctx.now.minute == 59, f"Expected minute=59 in context.now, got {ctx.now}"
            assert ctx.now.second == 59, f"Expected second=59 in context.now, got {ctx.now}"
            assert ctx.now.weekday() == 4, f"Expected Friday (weekday=4), got {ctx.now}"

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_replay_snapshot_date_used_in_get_completed_weekly_bars_has_fixed_time(
        self, mock_bs, mock_config, mock_stocks,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write, mock_ensure,
        mock_exists, mock_getsize,
    ):
        """The now value that flows into get_completed_weekly_bars (via
        context.now from the snapshot_date) must carry a fixed time >= 20:00
        so that the cutoff logic is stable.  We prove this by intercepting
        get_completed_weekly_bars and asserting the hour on its `now` param."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        captured_now_values = []

        def fake_scan(symbol, df, context, **kwargs):
            # Simulate what the real strategy does: call get_completed_weekly_bars
            import data_utils
            result = data_utils.get_completed_weekly_bars(df, now=context.now)
            captured_now_values.append(context.now)
            return _make_matched_result()

        strategy.scan.side_effect = fake_scan

        with patch("scanner_app.generate_weekly_snapshot_dates") as mock_dates:
            mock_dates.return_value = [
                datetime(2025, 5, 2, 23, 59, 59),
                datetime(2025, 5, 9, 23, 59, 59),
            ]
            run_weekly_replay_validation()

        assert len(captured_now_values) == 2
        for now_val in captured_now_values:
            assert now_val.hour >= 20, (
                f"context.now.hour must be >= 20 for stable weekly cutoff, "
                f"got hour={now_val.hour} for {now_val}"
            )
            assert now_val.weekday() == 4, f"Expected Friday (weekday=4), got {now_val}"

    # ------------------------------------------------------------------
    # lookback_weeks cardinality (Problem 1 fix)
    # ------------------------------------------------------------------

    def test_lookback_weeks_52_returns_exactly_52_snapshots(self):
        """lookback_weeks=52 must return exactly 52 snapshots, with the
        last being the last completed Friday anchored at 23:59:59, and
        the first being exactly 51 weeks before it."""
        mock_dt = self._make_friday_mock(hour=21)
        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-05-01")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(52)

        assert len(dates) == 52, f"Expected 52 snapshots, got {len(dates)}"

        expected_last = datetime(2026, 5, 1, 23, 59, 59)
        assert dates[-1] == expected_last, \
            f"Expected last snapshot {expected_last}, got {dates[-1]}"

        # First snapshot = last_completed_friday - 51 weeks (not 52)
        expected_first = expected_last - timedelta(weeks=51)
        assert dates[0] == expected_first, \
            f"Expected first snapshot {expected_first}, got {dates[0]}"

        for d in dates:
            assert d.weekday() == 4, f"Expected Friday, got {d}"
            assert d.hour == 23 and d.minute == 59 and d.second == 59, \
                f"Expected fixed time 23:59:59, got {d.time()}"

    def test_lookback_weeks_1_returns_single_snapshot(self):
        """Edge case: lookback_weeks=1 returns exactly the last completed Friday."""
        mock_dt = self._make_friday_mock(hour=21)
        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-05-01")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(1)

        assert len(dates) == 1, f"Expected 1 snapshot, got {len(dates)}"
        assert dates[0] == datetime(2026, 5, 1, 23, 59, 59)

    def test_lookback_weeks_4_returns_exactly_4_snapshots(self):
        """lookback_weeks=4 must return exactly 4 snapshots."""
        mock_dt = self._make_friday_mock(hour=21)
        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-05-01")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        assert len(dates) == 4, f"Expected 4 snapshots, got {len(dates)}"

        expected_dates = [
            datetime(2026, 4, 10, 23, 59, 59),
            datetime(2026, 4, 17, 23, 59, 59),
            datetime(2026, 4, 24, 23, 59, 59),
            datetime(2026, 5,  1, 23, 59, 59),
        ]
        for i, (actual, expected) in enumerate(zip(dates, expected_dates)):
            assert actual == expected, \
                f"Snapshot {i}: expected {expected}, got {actual}"

    # ------------------------------------------------------------------
    # Holiday short-week boundary tests (completed-week fix)
    # ------------------------------------------------------------------

    def test_holiday_short_week_after_last_trading_day_includes_current_week(self):
        """节假日短周：当前日期在最后交易日之后，应包含本周。

        Example: 2026-05-01 is a market holiday.  The last trading day of
        the week is 2026-04-30 (Thu).  Running replay on 2026-05-01 10:00
        must include the week ending 2026-05-01.
        """
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 5, 1, 10, 0, 0)
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-04-30")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        assert last == datetime(2026, 5, 1, 23, 59, 59), (
            f"Expected last snapshot 2026-05-01 23:59:59, got {last}"
        )

    def test_holiday_short_week_last_trading_day_before_20h_excludes_current_week(self):
        """节假日短周：当前日期等于最后交易日且未到 20:00，不应包含本周。

        Example: 2026-04-30 is the last trading day.  Running at 15:00 on
        that day must exclude the current week.
        """
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 4, 30, 15, 0, 0)
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-04-30")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        expected_last = datetime(2026, 4, 24, 23, 59, 59)
        assert last == expected_last, (
            f"Expected last snapshot {expected_last}, got {last}"
        )

    def test_holiday_short_week_last_trading_day_after_20h_includes_current_week(self):
        """节假日短周：当前日期等于最后交易日且已过 20:00，应包含本周.

        Example: 2026-04-30 is the last trading day.  Running at 21:00 on
        that day must include the current week.
        """
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 4, 30, 21, 0, 0)
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.get_last_trading_day_of_week", return_value=pd.Timestamp("2026-04-30")):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        assert last == datetime(2026, 5, 1, 23, 59, 59), (
            f"Expected last snapshot 2026-05-01 23:59:59, got {last}"
        )

    def test_calendar_query_failure_falls_back_conservatively(self):
        """交易日历查询失败时保守回退到上周 Friday anchor。"""
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 5, 1, 10, 0, 0)
        mock_dt.combine = datetime.combine
        mock_dt.min = datetime.min

        with patch("scanner_app.get_last_trading_day_of_week", return_value=None):
            with patch("scanner_app.datetime", mock_dt):
                dates = generate_weekly_snapshot_dates(4)

        last = dates[-1]
        expected_last = datetime(2026, 4, 24, 23, 59, 59)
        assert last == expected_last, (
            f"Expected last snapshot {expected_last} after calendar failure, got {last}"
        )


# ---------------------------------------------------------------------------
# Step 5-prep: Snapshot window identity validation in checkpoint
# ---------------------------------------------------------------------------

class TestSnapshotWindowIdentity:
    """Verify that resume only proceeds when the checkpoint's snapshot
    window matches the current run's snapshot window."""

    def _base_config(self):
        return {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_same_snapshot_window_allows_resume(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_results, mock_ensure,
        mock_exists, mock_getsize, mock_save_checkpoint, mock_load_checkpoint,
    ):
        """When checkpoint snapshot_dates matches the current run's
        generated dates exactly, resume must work: completed snapshots
        are skipped, only pending run."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]

        # Same window as stored in checkpoint
        mock_dates.return_value = [
            datetime(2025, 5, 2, 23, 59, 59),
            datetime(2025, 5, 9, 23, 59, 59),
            datetime(2025, 5, 16, 23, 59, 59),
        ]
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-10",  # 2025-05-16 + 21 weeks
        }

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect

        run_weekly_replay_validation()

        # Only pending snapshots processed: 05-09 and 05-16
        assert strategy.scan.call_count == 2
        scanned_dates = [call.args[2].now.strftime("%Y-%m-%d") for call in strategy.scan.call_args_list]
        assert "2025-05-02" not in scanned_dates, "05-02 must be skipped (already completed)"
        assert "2025-05-09" in scanned_dates
        assert "2025-05-16" in scanned_dates

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    def test_different_snapshot_window_rejects_resume(
        self, mock_bs, mock_create, mock_config, mock_stocks, mock_dates,
        mock_ensure, mock_exists, mock_load_checkpoint,
    ):
        """When checkpoint snapshot_dates differs from the current run's
        window, resume must be rejected with a clear RuntimeError."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_create.return_value = MagicMock()

        # Current run generates a *different* window
        mock_dates.return_value = [
            datetime(2026, 4, 3, 23, 59, 59),
            datetime(2026, 4, 10, 23, 59, 59),
            datetime(2026, 4, 17, 23, 59, 59),
        ]
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": ["2025-05-02"],
        }

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect

        with pytest.raises(RuntimeError) as exc_info:
            run_weekly_replay_validation()

        message = str(exc_info.value)
        assert "snapshot window" in message.lower(), \
            f"Error must mention snapshot window: {message}"
        assert "cannot safely resume" in message.lower(), \
            f"Error must mention cannot resume: {message}"
        # Must include both windows for diagnostics
        assert "2025-05-02" in message, \
            f"Error must include checkpoint window start: {message}"
        assert "2026-04-03" in message, \
            f"Error must include current window start: {message}"

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.os.path.exists")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    def test_old_checkpoint_missing_snapshot_dates_rejects_resume(
        self, mock_bs, mock_create, mock_config, mock_stocks, mock_dates,
        mock_ensure, mock_exists, mock_load_checkpoint,
    ):
        """A checkpoint created before the snapshot_dates field existed
        must be rejected — even if experiment_tag/universe/lookback_weeks/
        version all match.  Conservative safety: no silent compatibility."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_create.return_value = MagicMock()

        mock_dates.return_value = [datetime(2025, 5, 2, 23, 59, 59)]
        # Old-format checkpoint: everything matches on experiment identity,
        # but snapshot_dates is absent
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "completed_snapshots": ["2025-05-02"],
        }

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")

        mock_exists.side_effect = exists_side_effect

        with pytest.raises(RuntimeError) as exc_info:
            run_weekly_replay_validation()

        message = str(exc_info.value)
        assert "snapshot_dates" in message.lower(), \
            f"Error must mention snapshot_dates: {message}"
        assert "cannot safely resume" in message.lower(), \
            f"Error must mention cannot safely resume: {message}"
        assert "insufficient" in message.lower() or "missing" in message.lower(), \
            f"Error must mention insufficient/missing schema: {message}"


# ---------------------------------------------------------------------------
# Step 5: Weekly-strategy replay optimization v1
# ---------------------------------------------------------------------------

class TestWeeklyReplayCachePath:
    """Verify that the weekly precompute cache is used for weekly strategies
    and that the optimization does not change business semantics.

    These tests cover:
    1. precompute_weekly_bars_for_replay is called once per symbol (not per snapshot)
    2. slice_weekly_bars_for_snapshot is called once per symbol per snapshot
    3. strategy.scan receives precomputed_weekly (not None) for weekly strategies
    4. Non-weekly strategies do NOT trigger the weekly cache path
    5. Fault tolerance: exception during precompute is isolated to that symbol
    6. Step 1-4 invariants (checkpoint, error CSV, done log) are preserved
    """

    def _base_config(self):
        return {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

    # ------------------------------------------------------------------
    # Test 1: precompute called once per symbol, not per snapshot
    # ------------------------------------------------------------------

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.0, "return_8w": 0.0, "return_12w": 0.0})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.slice_weekly_bars_for_snapshot")
    @patch("scanner_app.precompute_weekly_bars_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_precompute_called_once_per_symbol_not_per_snapshot(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_precompute, mock_slice,
        mock_load_full, mock_load_hist, mock_calc_returns,
        mock_write, mock_ensure, mock_exists, mock_getsize,
    ):
        """precompute_weekly_bars_for_replay must be called once per symbol
        regardless of how many snapshots there are."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000", "sh.600001"]
        # 3 snapshots, 2 symbols → precompute should be called 2 times total
        mock_dates.return_value = [
            datetime(2025, 5, 2, 23, 59, 59),
            datetime(2025, 5, 9, 23, 59, 59),
            datetime(2025, 5, 16, 23, 59, 59),
        ]

        weekly_df = _make_df(60)
        mock_precompute.return_value = weekly_df
        mock_slice.return_value = weekly_df

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        run_weekly_replay_validation()

        # 2 symbols × 1 precompute each = 2 total calls
        assert mock_precompute.call_count == 2, (
            f"Expected precompute called once per symbol (2), got {mock_precompute.call_count}"
        )
        # 2 symbols × 3 snapshots = 6 slice calls
        assert mock_slice.call_count == 6, (
            f"Expected slice called once per symbol per snapshot (6), got {mock_slice.call_count}"
        )

    # ------------------------------------------------------------------
    # Test 2: strategy.scan receives precomputed_weekly kwarg
    # ------------------------------------------------------------------

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.0, "return_8w": 0.0, "return_12w": 0.0})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.slice_weekly_bars_for_snapshot")
    @patch("scanner_app.precompute_weekly_bars_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_scan_receives_precomputed_weekly_kwarg(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_precompute, mock_slice,
        mock_load_full, mock_load_hist, mock_calc_returns,
        mock_write, mock_ensure, mock_exists, mock_getsize,
    ):
        """strategy.scan must be called with precomputed_weekly= for weekly strategies."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 9, 23, 59, 59)]

        sentinel_weekly = _make_df(30)
        mock_precompute.return_value = sentinel_weekly
        mock_slice.return_value = sentinel_weekly

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        run_weekly_replay_validation()

        assert strategy.scan.call_count == 1
        call_kwargs = strategy.scan.call_args.kwargs
        assert "precomputed_weekly" in call_kwargs, (
            f"Expected precomputed_weekly kwarg in scan call, got kwargs={call_kwargs}"
        )
        assert call_kwargs["precomputed_weekly"] is sentinel_weekly, (
            "precomputed_weekly must be the slice returned by slice_weekly_bars_for_snapshot"
        )

    # ------------------------------------------------------------------
    # Test 3: non-weekly strategy does NOT use weekly cache
    # ------------------------------------------------------------------

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.0, "return_8w": 0.0, "return_12w": 0.0})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.slice_weekly_bars_for_snapshot")
    @patch("scanner_app.precompute_weekly_bars_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_non_weekly_strategy_does_not_use_weekly_cache(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_precompute, mock_slice,
        mock_load_full, mock_load_hist, mock_calc_returns,
        mock_write, mock_ensure, mock_exists, mock_getsize,
    ):
        """A strategy not in _WEEKLY_REPLAY_STRATEGIES must not trigger
        precompute or slice, and scan must be called without precomputed_weekly."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 9, 23, 59, 59)]

        strategy = MagicMock()
        # A hypothetical daily strategy name not in _WEEKLY_REPLAY_STRATEGIES
        strategy.name = "moving_average"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        run_weekly_replay_validation()

        mock_precompute.assert_not_called()
        mock_slice.assert_not_called()
        # scan must be called without precomputed_weekly kwarg
        assert strategy.scan.call_count == 1
        call_kwargs = strategy.scan.call_args.kwargs
        assert "precomputed_weekly" not in call_kwargs, (
            f"Non-weekly strategy must not receive precomputed_weekly, got {call_kwargs}"
        )

    # ------------------------------------------------------------------
    # Test 4: precompute exception isolates to that symbol
    # ------------------------------------------------------------------

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.0, "return_8w": 0.0, "return_12w": 0.0})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.precompute_weekly_bars_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_precompute_exception_isolates_to_symbol(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_precompute,
        mock_load_full, mock_load_hist, mock_calc_returns,
        mock_write_errors, mock_write_results,
        mock_ensure, mock_exists, mock_getsize, caplog,
    ):
        """If precompute_weekly_bars_for_replay raises for one symbol,
        that symbol is recorded as failed and the snapshot continues."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.000001", "sh.000002", "sh.000003"]
        mock_dates.return_value = [datetime(2025, 5, 9, 23, 59, 59)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        def precompute_side_effect(full_df):
            # Raise only for the second symbol (detected via call count)
            if mock_precompute.call_count == 2:
                raise RuntimeError("Mocked precompute failure")
            return _make_df(30)

        mock_precompute.side_effect = precompute_side_effect

        with caplog.at_level(logging.ERROR):
            run_weekly_replay_validation()

        # One error logged for sh.000002
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and "failed at stage" in r.message]
        assert len(error_logs) == 1, f"Expected 1 error log, got {len(error_logs)}"
        assert "sh.000002" in error_logs[0].message

        # Error CSV written; results CSV written for the two successful symbols
        mock_write_errors.assert_called_once()
        mock_write_results.assert_called_once()

    # ------------------------------------------------------------------
    # Test 5: checkpoint/resume semantics preserved with weekly cache
    # ------------------------------------------------------------------

    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.0, "return_8w": 0.0, "return_12w": 0.0})
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.slice_weekly_bars_for_snapshot")
    @patch("scanner_app.precompute_weekly_bars_for_replay")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.load_config")
    @patch("scanner_app.bs")
    def test_checkpoint_saved_after_each_snapshot_with_weekly_cache(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_precompute, mock_slice,
        mock_load_full, mock_load_hist, mock_calc_returns,
        mock_write, mock_ensure, mock_exists, mock_getsize,
        mock_save_checkpoint,
    ):
        """Checkpoint must still be saved after each snapshot when weekly
        cache is active — Step 3 resume semantics must not be broken."""
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = self._base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [
            datetime(2025, 5, 2, 23, 59, 59),
            datetime(2025, 5, 9, 23, 59, 59),
        ]

        weekly_df = _make_df(60)
        mock_precompute.return_value = weekly_df
        mock_slice.return_value = weekly_df

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_unmatched_result()
        mock_create.return_value = strategy

        run_weekly_replay_validation()

        # One checkpoint save per snapshot
        assert mock_save_checkpoint.call_count == 2, (
            f"Expected 2 checkpoint saves (one per snapshot), got {mock_save_checkpoint.call_count}"
        )
        # Second save must include both snapshots
        final_completed = mock_save_checkpoint.call_args_list[-1].args[5]
        assert "2025-05-02" in final_completed
        assert "2025-05-09" in final_completed


# ---------------------------------------------------------------------------
# Step 5: Unit tests for the helper functions themselves
# ---------------------------------------------------------------------------

class TestWeeklyReplayHelpers:
    """Unit tests for precompute_weekly_bars_for_replay and
    slice_weekly_bars_for_snapshot in isolation."""

    def _make_daily_df(self, start: str = "2025-01-06", weeks: int = 20) -> pd.DataFrame:
        """Build a daily DataFrame spanning `weeks` full Mon-Fri weeks."""
        dates = pd.bdate_range(start=start, periods=weeks * 5)
        n = len(dates)
        return pd.DataFrame({
            "date": dates,
            "open": [10.0] * n,
            "high": [10.5] * n,
            "low": [9.5] * n,
            "close": [10.0 + i * 0.01 for i in range(n)],
            "volume": [1000] * n,
        })

    def test_precompute_returns_friday_anchored_weekly_bars(self):
        """precompute_weekly_bars_for_replay must return W-FRI weekly bars."""
        from scanner_app import precompute_weekly_bars_for_replay
        daily = self._make_daily_df(weeks=8)
        weekly = precompute_weekly_bars_for_replay(daily)

        assert not weekly.empty
        # All dates must be Fridays (weekday 4)
        for d in pd.to_datetime(weekly["date"]):
            assert d.weekday() == 4, f"Expected Friday, got {d} (weekday={d.weekday()})"

    def test_slice_returns_bars_up_to_snapshot_friday(self):
        """slice_weekly_bars_for_snapshot must return only bars on or before
        the snapshot's Friday anchor."""
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        daily = self._make_daily_df(start="2025-01-06", weeks=12)
        weekly = precompute_weekly_bars_for_replay(daily)

        # Snapshot = Friday 2025-02-28 at 23:59:59
        snapshot = datetime(2025, 2, 28, 23, 59, 59)
        sliced = slice_weekly_bars_for_snapshot(weekly, snapshot)

        assert not sliced.empty
        cutoff = pd.Timestamp("2025-02-28")
        for d in pd.to_datetime(sliced["date"]):
            assert d <= cutoff, f"Sliced bar {d} is after cutoff {cutoff}"

    def test_slice_semantic_equivalence_with_get_completed_weekly_bars(self):
        """The precompute+slice path must produce the same weekly bars as
        calling get_completed_weekly_bars directly on the daily slice.

        This is the core correctness guarantee for the optimization.

        We mock get_last_trading_day_of_week so the standard path doesn't
        need a live baostock connection — the mock returns the Friday itself,
        which is what a normal trading week would return.
        """
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        daily = self._make_daily_df(start="2025-01-06", weeks=16)

        # Snapshot = Friday 2025-03-28 at 23:59:59 (fixed time anchor)
        snapshot = datetime(2025, 3, 28, 23, 59, 59)
        cutoff_ts = pd.Timestamp(snapshot.date())

        # Path A: precompute + slice (new optimization path)
        weekly_full = precompute_weekly_bars_for_replay(daily)
        # Pass the snapshot week's last trading day so the completeness guard
        # keeps the bar (this is a normal 5-day week, data is complete).
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=pd.Timestamp("2025-03-28"),
        )

        # Path B: standard path — filter daily to cutoff, then get_completed_weekly_bars.
        # Mock get_last_trading_day_of_week so it returns the Friday itself (normal week),
        # avoiding a live baostock call in the test environment.
        daily_slice = daily[pd.to_datetime(daily["date"]) <= cutoff_ts].copy()
        with patch("data_utils.get_last_trading_day_of_week", return_value=pd.Timestamp("2025-03-28")):
            standard = get_completed_weekly_bars(daily_slice, now=snapshot)

        assert len(sliced) == len(standard), (
            f"Row count mismatch: precompute+slice={len(sliced)}, standard={len(standard)}"
        )
        if not sliced.empty and not standard.empty:
            pd.testing.assert_frame_equal(
                sliced.reset_index(drop=True),
                standard.reset_index(drop=True),
                check_dtype=False,
            )

    def test_slice_empty_weekly_returns_empty(self):
        """slice_weekly_bars_for_snapshot on an empty DataFrame returns empty."""
        from scanner_app import slice_weekly_bars_for_snapshot
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        result = slice_weekly_bars_for_snapshot(empty, datetime(2025, 5, 9, 23, 59, 59))
        assert result.empty


# ---------------------------------------------------------------------------
# Step 5 patch: MomentumReversal13Strategy.scan precomputed_weekly compat
# ---------------------------------------------------------------------------

class TestMR13PrecomputedWeeklyCompat:
    """Verify that MomentumReversal13Strategy.scan accepts the precomputed_weekly
    kwarg introduced by the weekly replay cache path (Step 5).

    Three cases:
    1. precomputed_weekly passed → no TypeError, result returned normally
    2. precomputed_weekly not None → get_completed_weekly_bars NOT called
    3. precomputed_weekly is None (default) → get_completed_weekly_bars IS called
    """

    def _make_weekly_df(self, n: int = 20) -> pd.DataFrame:
        fridays = pd.date_range("2025-01-10", periods=n, freq="W-FRI")
        return pd.DataFrame({
            "date": fridays,
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0 + i * 0.05 for i in range(n)],
            "volume": [5000] * n,
            "turn": [1.0] * n,
            "pctChg": [0.5] * n,
        })

    def _make_daily_df(self, n: int = 100) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-06", periods=n)
        return pd.DataFrame({
            "date": dates,
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0 + i * 0.01 for i in range(n)],
            "volume": [5000] * n,
            "turn": [1.0] * n,
            "pctChg": [0.5] * n,
        })

    def _make_context(self) -> StrategyContext:
        return StrategyContext(
            now=datetime(2025, 5, 9, 23, 59, 59),
            stock_pool=[],
            config={},
        )

    def test_scan_accepts_precomputed_weekly_no_type_error(self):
        """scan() must not raise TypeError when called with precomputed_weekly kwarg."""
        from strategies.momentum_reversal_13 import MomentumReversal13Strategy
        strategy = MomentumReversal13Strategy({})
        weekly = self._make_weekly_df(20)
        daily = self._make_daily_df()
        ctx = self._make_context()

        # Must not raise — this is the exact call the replay loop makes
        result = strategy.scan("sh.600000", daily, ctx, precomputed_weekly=weekly)
        assert result is not None
        assert hasattr(result, "matched")

    def test_scan_with_precomputed_weekly_skips_get_completed_weekly_bars(self):
        """When precomputed_weekly is provided, get_completed_weekly_bars must
        not be called — the strategy uses the pre-sliced data directly."""
        from strategies.momentum_reversal_13 import MomentumReversal13Strategy
        strategy = MomentumReversal13Strategy({})
        weekly = self._make_weekly_df(20)
        daily = self._make_daily_df()
        ctx = self._make_context()

        with patch("strategies.momentum_reversal_13.get_completed_weekly_bars") as mock_gcwb:
            strategy.scan("sh.600000", daily, ctx, precomputed_weekly=weekly)
            mock_gcwb.assert_not_called()

    def test_scan_without_precomputed_weekly_calls_get_completed_weekly_bars(self):
        """When precomputed_weekly is None (default), get_completed_weekly_bars
        must still be called — the normal scanner path is unchanged."""
        from strategies.momentum_reversal_13 import MomentumReversal13Strategy
        strategy = MomentumReversal13Strategy({})
        daily = self._make_daily_df()
        ctx = self._make_context()

        with patch("strategies.momentum_reversal_13.get_completed_weekly_bars",
                   return_value=self._make_weekly_df(20)) as mock_gcwb:
            strategy.scan("sh.600000", daily, ctx)
            mock_gcwb.assert_called_once()


# ---------------------------------------------------------------------------
# Weekly replay cache completeness guard — symbol-level data-integrity tests
#
# These tests verify that slice_weekly_bars_for_snapshot NEVER produces a
# "half-week" phantom bar for the current snapshot week when the symbol's
# full_df does not contain all trading days through the week's last trading
# day.  The cache path must be no looser than the standard
# get_completed_weekly_bars path.
# ---------------------------------------------------------------------------

class TestWeeklyReplayCompletenessGuard:
    """Verify that slice_weekly_bars_for_snapshot enforces symbol-level
    completed-week data-integrity via the new trading_days_count and
    last_daily_date fields."""

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _daily_with_last_week_partial(
        start: str = "2025-01-06",
        full_weeks: int = 8,
        last_week_dates: list | None = None,
    ) -> pd.DataFrame:
        """Build a daily DataFrame with `full_weeks` complete Mon-Fri weeks
        plus a partial final week specified by `last_week_dates` (list of
        date strings or timestamps)."""
        full_dates = pd.bdate_range(start=start, periods=full_weeks * 5)
        rows = []
        for d in full_dates:
            rows.append({
                "date": d,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1000,
            })
        if last_week_dates:
            for d in last_week_dates:
                rows.append({
                    "date": pd.Timestamp(d),
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1000,
                })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _last_bar_date(weekly: pd.DataFrame) -> pd.Timestamp | None:
        if weekly.empty:
            return None
        return pd.Timestamp(weekly["date"].iloc[-1])

    @staticmethod
    def _assert_identical(sliced: pd.DataFrame, standard: pd.DataFrame):
        """Assert two weekly DataFrames are identical in content and shape."""
        assert len(sliced) == len(standard), (
            f"Row count mismatch: sliced={len(sliced)}, standard={len(standard)}"
        )
        if not sliced.empty and not standard.empty:
            pd.testing.assert_frame_equal(
                sliced.reset_index(drop=True),
                standard.reset_index(drop=True),
                check_dtype=False,
            )

    # -- test cases -------------------------------------------------------

    def test_normal_complete_week_includes_snapshot_bar_and_matches_standard(self):
        """Normal 5-day week with full daily data: cache path must include
        the snapshot-week bar and produce identical results to the standard
        get_completed_weekly_bars path."""
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        # 10 full Mon-Fri weeks ending 2025-03-14 (Friday)
        daily = self._daily_with_last_week_partial(
            start="2025-01-06", full_weeks=10, last_week_dates=None,
        )
        snapshot = datetime(2025, 3, 14, 23, 59, 59)
        cutoff_ts = pd.Timestamp(snapshot.date())

        # Cache path
        weekly_full = precompute_weekly_bars_for_replay(daily)
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=pd.Timestamp("2025-03-14"),
        )

        # Standard path (mock calendar so it returns the Friday itself)
        daily_slice = daily[pd.to_datetime(daily["date"]) <= cutoff_ts].copy()
        with patch("data_utils.get_last_trading_day_of_week", return_value=pd.Timestamp("2025-03-14")):
            standard = get_completed_weekly_bars(daily_slice, now=snapshot)

        # Both must include the snapshot-week bar (2025-03-14)
        assert self._last_bar_date(sliced) == pd.Timestamp("2025-03-14")
        assert self._last_bar_date(standard) == pd.Timestamp("2025-03-14")
        self._assert_identical(sliced, standard)

    def test_holiday_short_week_thursday_last_trading_day_keeps_bar(self):
        """Week where Friday is a holiday (last trading day = Thursday) and
        symbol data reaches Thursday: cache path must keep the bar and
        match the standard path."""
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        snapshot = datetime(2025, 5, 2, 23, 59, 59)  # Friday, but May 1-2 are holidays
        cutoff_ts = pd.Timestamp(snapshot.date())

        # 8 full weeks + final week Mon-Thu only (no Friday — holiday)
        daily = self._daily_with_last_week_partial(
            start="2025-03-03", full_weeks=8,
            last_week_dates=["2025-04-28", "2025-04-29", "2025-04-30"],
        )

        weekly_full = precompute_weekly_bars_for_replay(daily)
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=pd.Timestamp("2025-04-30"),  # Thursday
        )

        daily_slice = daily[pd.to_datetime(daily["date"]) <= cutoff_ts].copy()
        with patch("data_utils.get_last_trading_day_of_week", return_value=pd.Timestamp("2025-04-30")):
            standard = get_completed_weekly_bars(daily_slice, now=snapshot)

        assert self._last_bar_date(sliced) == pd.Timestamp("2025-05-02")  # Friday anchor
        assert self._last_bar_date(standard) == pd.Timestamp("2025-05-02")
        self._assert_identical(sliced, standard)

    def test_normal_week_data_only_to_wednesday_drops_snapshot_bar(self):
        """Full-trading-week but symbol's full_df only contains Mon-Wed:
        cache path must NOT include a half-week "phantom" bar anchored at
        Friday."""
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        snapshot = datetime(2025, 3, 14, 23, 59, 59)
        cutoff_ts = pd.Timestamp(snapshot.date())

        daily = self._daily_with_last_week_partial(
            start="2025-01-06", full_weeks=9,
            last_week_dates=["2025-03-10", "2025-03-11", "2025-03-12"],
        )

        weekly_full = precompute_weekly_bars_for_replay(daily)
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=pd.Timestamp("2025-03-14"),  # Friday
        )

        # Cache path must NOT include the 2025-03-14 bar
        assert self._last_bar_date(sliced) != pd.Timestamp("2025-03-14"), (
            "Cache path must drop half-week phantom bar for incomplete data"
        )

        # Standard path: should also drop the bar since data doesn't reach last trading day
        daily_slice = daily[pd.to_datetime(daily["date"]) <= cutoff_ts].copy()
        with patch("data_utils.get_last_trading_day_of_week", return_value=pd.Timestamp("2025-03-14")):
            standard = get_completed_weekly_bars(daily_slice, now=snapshot)

        self._assert_identical(sliced, standard)

    def test_short_week_with_only_2_trading_days_drops_bar(self):
        """Week with a holiday-shortened schedule (only 2 trading days)
        AND data covers both days: trading_days_count=2 < 3 threshold
        forces the bar to be dropped.

        Note: the standard path (get_completed_weekly_bars) does NOT have
        a trading_days_count guard, so it may still include the bar.
        The cache path may be *stricter* but must never be *looser*.
        """
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        snapshot = datetime(2025, 10, 3, 23, 59, 59)  # National Day week
        cutoff_ts = pd.Timestamp(snapshot.date())

        # Only Mon (Sep 29) and Tue (Sep 30) — Wed–Fri are holidays
        daily = self._daily_with_last_week_partial(
            start="2025-07-28", full_weeks=8,
            last_week_dates=["2025-09-29", "2025-09-30"],
        )

        weekly_full = precompute_weekly_bars_for_replay(daily)
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=pd.Timestamp("2025-09-30"),
        )

        # Cache path must drop the bar: trading_days_count=2 < 3
        assert self._last_bar_date(sliced) != pd.Timestamp("2025-10-03"), (
            "Cache path must drop bar when trading_days_count < 3"
        )

        # Verify cache path is NOT looser than standard path:
        # the standard path may include bars the cache path drops,
        # but the cache path must never include bars the standard
        # path does not.
        daily_slice = daily[pd.to_datetime(daily["date"]) <= cutoff_ts].copy()
        with patch("data_utils.get_last_trading_day_of_week", return_value=pd.Timestamp("2025-09-30")):
            standard = get_completed_weekly_bars(daily_slice, now=snapshot)

        standard_dates = set(pd.to_datetime(standard["date"]))
        sliced_dates = set(pd.to_datetime(sliced["date"]))
        extra_in_cache = sliced_dates - standard_dates
        assert not extra_in_cache, (
            f"Cache path is looser than standard path! Extra bars: {extra_in_cache}"
        )

    def test_calendar_query_failure_drops_snapshot_bar_conservatively(self):
        """When the trading calendar query fails (last_trading_day=None),
        the cache path must conservatively drop the snapshot-week bar."""
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot

        daily = self._daily_with_last_week_partial(
            start="2025-01-06", full_weeks=10, last_week_dates=None,
        )
        snapshot = datetime(2025, 3, 14, 23, 59, 59)

        weekly_full = precompute_weekly_bars_for_replay(daily)
        sliced = slice_weekly_bars_for_snapshot(
            weekly_full, snapshot,
            snapshot_week_last_trading_day=None,  # calendar failure
        )

        assert self._last_bar_date(sliced) != pd.Timestamp("2025-03-14"), (
            "Calendar failure must conservatively drop snapshot-week bar"
        )

    def test_cross_path_comparison_varied_scenarios(self):
        """For the same daily data, the precompute+slice cache path and the
        standard get_completed_weekly_bars path must produce identical
        results across varied scenarios:
          - normal complete week at snapshot
          - holiday-shortened week with sufficient data
          - partial data (Mon-Wed) in a normal week
        """
        from scanner_app import precompute_weekly_bars_for_replay, slice_weekly_bars_for_snapshot
        from data_utils import get_completed_weekly_bars

        scenarios = [
            # (label, start, full_weeks, last_week_dates, snapshot,
            #  last_trading_day_mock, snapshot_week_last_trading_day)
            {
                "label": "normal complete week",
                "start": "2025-01-06",
                "full_weeks": 10,
                "last_week_dates": None,
                "snapshot": datetime(2025, 3, 14, 23, 59, 59),
                "last_trading_day_mock": pd.Timestamp("2025-03-14"),
                "snapshot_week_last_trading_day": pd.Timestamp("2025-03-14"),
            },
            {
                "label": "holiday Thursday close",
                "start": "2025-03-03",
                "full_weeks": 8,
                "last_week_dates": ["2025-04-28", "2025-04-29", "2025-04-30"],
                "snapshot": datetime(2025, 5, 2, 23, 59, 59),
                "last_trading_day_mock": pd.Timestamp("2025-04-30"),
                "snapshot_week_last_trading_day": pd.Timestamp("2025-04-30"),
            },
            {
                "label": "partial data Mon-Wed",
                "start": "2025-01-06",
                "full_weeks": 9,
                "last_week_dates": ["2025-03-10", "2025-03-11", "2025-03-12"],
                "snapshot": datetime(2025, 3, 14, 23, 59, 59),
                "last_trading_day_mock": pd.Timestamp("2025-03-14"),
                "snapshot_week_last_trading_day": pd.Timestamp("2025-03-14"),
            },
        ]

        for sc in scenarios:
            daily = self._daily_with_last_week_partial(
                start=sc["start"],
                full_weeks=sc["full_weeks"],
                last_week_dates=sc["last_week_dates"],
            )

            weekly_full = precompute_weekly_bars_for_replay(daily)
            sliced = slice_weekly_bars_for_snapshot(
                weekly_full, sc["snapshot"],
                snapshot_week_last_trading_day=sc["snapshot_week_last_trading_day"],
            )

            daily_slice = daily[
                pd.to_datetime(daily["date"]) <= pd.Timestamp(sc["snapshot"].date())
            ].copy()
            with patch("data_utils.get_last_trading_day_of_week",
                       return_value=sc["last_trading_day_mock"]):
                standard = get_completed_weekly_bars(daily_slice, now=sc["snapshot"])

            self._assert_identical(sliced, standard)


# replay_data_end_date tests have been moved to test_replay_data_end_date.py
# to avoid mock issues in this file.


