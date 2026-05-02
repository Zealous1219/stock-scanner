"""Tests for failed_symbols_per_snapshot in weekly replay.

Covers:
- Partial failure: snapshot marked completed, failed symbols recorded
- No-failure clears stale failed-symbol records
- Legacy checkpoint without field is compatible
- Completed snapshot with failed symbols is valid checkpoint state
- Resume skips completed snapshot even with failed symbols, logs warning
- Type validation on failed_symbols_per_snapshot raises RuntimeError
- Replay end logs unresolved failure summary
"""

import logging
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

from scanner_app import (
    run_weekly_replay_validation,
    validate_replay_checkpoint,
    _summarize_failed_symbols,
)
from strategy_runtime import StrategyDecision, StrategyResult, StrategyContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(days: int = 120):
    import pandas as pd
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


def _make_matched_result():
    return StrategyResult(
        matched=True,
        reason_code="signal",
        reason_text="matched",
        details={"signal_date": "2025-06-01"},
    )


def _base_config():
    return {
        "strategy": {"params": {}},
        "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
    }


# ---------------------------------------------------------------------------
# Test 1: partial failure snapshot still marked completed
# ---------------------------------------------------------------------------

class TestPartialFailureStillCompleted:
    """A snapshot where some symbols fail must still be marked as completed,
    and the failed symbols must appear in failed_symbols_per_snapshot."""

    @patch("scanner_app.os.path.getsize", return_value=0)
    @patch("scanner_app.os.path.exists", return_value=False)
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.save_replay_checkpoint")
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
    def test_partial_failure_records_failed_symbols(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results,
        mock_save_checkpoint, mock_ensure,
        mock_exists, mock_getsize,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = _base_config()
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        # Two symbols: one succeeds, one fails
        mock_stocks.return_value = ["sh.600000", "sh.600001"]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")

        # sh.600001 fails in ALL snapshots; sh.600000 always succeeds
        def scan_side_effect(symbol, df, context, **kwargs):
            if symbol == "sh.600001":
                raise ValueError("simulated scan error")
            return _make_matched_result()

        strategy.scan.side_effect = scan_side_effect
        mock_create.return_value = strategy

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                run_weekly_replay_validation()

        assert mock_save_checkpoint.call_count == 2

        # Check the last save — both snapshots are completed and both have
        # sh.600001 in failed_symbols_per_snapshot
        last_call = mock_save_checkpoint.call_args_list[-1]
        args = last_call.args
        kwargs = last_call.kwargs

        # completed_snapshots should include both snapshots
        assert args[5] == ["2025-05-02", "2025-05-09"]

        # failed_symbols_per_snapshot should contain both snapshots with ["sh.600001"]
        failed_map = kwargs.get("failed_symbols_per_snapshot", {})
        assert failed_map == {
            "2025-05-02": ["sh.600001"],
            "2025-05-09": ["sh.600001"],
        }


# ---------------------------------------------------------------------------
# Test 2: snapshot with no failures clears stale failed-symbol record
# ---------------------------------------------------------------------------

class TestClearStaleFailureRecord:
    """When a snapshot that previously had failed symbols is re-run (because
    it was NOT yet marked as completed) and succeeds completely, the stale
    failed_symbols_per_snapshot entry must be removed."""

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
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
    def test_clear_stale_when_no_failures(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results,
        mock_ensure,
        mock_exists, mock_getsize, mock_save_checkpoint, mock_load_checkpoint,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = _base_config()
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9), datetime(2025, 5, 16)]

        mock_stocks.return_value = ["sh.600000"]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # 2025-05-02 is NOT yet completed but has a stale failed_symbols record
        # (for example, a previous interrupted run had failures but didn't complete).
        # Now on this run, it gets processed, finds no failures, and clears the stale entry.
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": [],
            "replay_data_end_date": "2025-10-10",  # 2025-05-16 + 21 weeks
            "failed_symbols_per_snapshot": {
                "2025-05-02": ["sh.600001"],  # stale — this snapshot will succeed now
            },
        }

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")
        mock_exists.side_effect = exists_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                run_weekly_replay_validation()

        # After all 3 snapshots processed with no failures, failed_symbols_per_snapshot should be empty
        last_call = mock_save_checkpoint.call_args_list[-1]
        kwargs = last_call.kwargs
        failed_map = kwargs.get("failed_symbols_per_snapshot", {})
        assert failed_map == {}


# ---------------------------------------------------------------------------
# Test 3: legacy checkpoint without failed_symbols_per_snapshot
# ---------------------------------------------------------------------------

class TestLegacyCheckpointCompatibility:
    """A checkpoint without failed_symbols_per_snapshot must be handled
    gracefully (treated as {})."""

    def test_legacy_checkpoint_validates(self):
        checkpoint = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",
        }
        # Should not raise
        result = validate_replay_checkpoint(
            checkpoint,
            "mr13_all_52w_v1",
            "all",
            52,
            "v1",
            ["2025-05-02", "2025-05-09"],
            expected_replay_data_end_date="2025-10-03",
        )
        assert result == ["2025-05-02"]

    def test_legacy_checkpoint_no_field_does_not_crash(self):
        """validate_replay_checkpoint should not care if the field is absent."""
        checkpoint = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",
        }
        validate_replay_checkpoint(
            checkpoint,
            "mr13_all_52w_v1", "all", 52, "v1",
            ["2025-05-02"],
            expected_replay_data_end_date="2025-10-03",
        )
        # No exception = pass


# ---------------------------------------------------------------------------
# Test 4: completed snapshot with failed symbols is valid
# ---------------------------------------------------------------------------

class TestCompletedWithFailedSymbolsIsValid:
    """A checkpoint where a completed snapshot also has failed symbols
    must NOT cause validate_replay_checkpoint to error."""

    def test_completed_with_failures_is_valid_state(self):
        checkpoint = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02", "2025-05-09"],
            "replay_data_end_date": "2025-10-03",
            "failed_symbols_per_snapshot": {
                "2025-05-02": ["sh.600001"],
            },
        }
        result = validate_replay_checkpoint(
            checkpoint,
            "mr13_all_52w_v1", "all", 52, "v1",
            ["2025-05-02", "2025-05-09"],
            expected_replay_data_end_date="2025-10-03",
        )
        # Both snapshots are completed — this is valid even with failed symbols
        assert result == ["2025-05-02", "2025-05-09"]

        # The validation should not check cross-consistency between completed_snapshots
        # and failed_symbols_per_snapshot — they are independent concerns.


# ---------------------------------------------------------------------------
# Test 5: resume still skips completed snapshot even with failed symbols
# ---------------------------------------------------------------------------

class TestResumeSkipsCompletedWithFailures:
    """Resume must skip completed snapshots regardless of failed_symbols_per_snapshot,
    and log a warning about unresolved failures."""

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
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
    def test_resume_skips_and_logs_warning(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results,
        mock_ensure, mock_exists, mock_getsize,
        mock_save_checkpoint, mock_load_checkpoint, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = _base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9), datetime(2025, 5, 16)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Both snapshots completed, but 2025-05-02 has failed symbols
        mock_load_checkpoint.return_value = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
            "completed_snapshots": ["2025-05-02", "2025-05-09"],
            "replay_data_end_date": "2025-10-10",  # 2025-05-16 + 21 weeks
            "failed_symbols_per_snapshot": {
                "2025-05-02": ["sh.600001", "sh.600002"],
                "2025-05-09": ["sh.600003"],
            },
        }

        def exists_side_effect(path):
            return path.endswith(".checkpoint.json") or path.endswith(".csv")
        mock_exists.side_effect = exists_side_effect

        with caplog.at_level(logging.WARNING):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("scanner_app.VALIDATION_DIR", tmpdir):
                    run_weekly_replay_validation()

        # Should have warnings about unresolved failures (at resume start AND at replay end)
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        unresolved_warnings = [r for r in warning_logs if "Unresolved failed symbols" in r.message]
        assert len(unresolved_warnings) == 2  # one at resume, one at end
        assert "2 snapshot(s) with 3 total failed symbol(s)" in unresolved_warnings[0].message

        # Only 2025-05-16 should be processed (the other 2 skipped)
        processed_dates = [call.args[2].now.strftime("%Y-%m-%d") for call in strategy.scan.call_args_list]
        assert processed_dates == ["2025-05-16"]

        # Save checkpoint should still have both completed + the new one
        last_call = mock_save_checkpoint.call_args_list[-1]
        assert last_call.args[5] == ["2025-05-02", "2025-05-09", "2025-05-16"]

        # failed_symbols_per_snapshot should be unchanged for old ones (resume doesn't re-process)
        failed_map = last_call.kwargs.get("failed_symbols_per_snapshot", {})
        assert "2025-05-02" in failed_map
        assert "2025-05-09" in failed_map


# ---------------------------------------------------------------------------
# Test 6: type validation on failed_symbols_per_snapshot
# ---------------------------------------------------------------------------

class TestFailedSymbolsTypeValidation:
    """Invalid types in failed_symbols_per_snapshot must raise RuntimeError."""

    def _valid_base_checkpoint(self):
        return {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",
        }

    def _validate(self, checkpoint):
        validate_replay_checkpoint(
            checkpoint,
            "mr13_all_52w_v1", "all", 52, "v1",
            ["2025-05-02", "2025-05-09"],
            expected_replay_data_end_date="2025-10-03",
        )

    def test_failed_symbols_not_a_dict(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = "not-a-dict"
        with pytest.raises(RuntimeError, match="failed_symbols_per_snapshot"):
            self._validate(checkpoint)

    def test_failed_symbols_key_not_string(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = {123: ["sh.600000"]}
        with pytest.raises(RuntimeError, match="all keys must be strings"):
            self._validate(checkpoint)

    def test_failed_symbols_value_not_list(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = {"2025-05-02": "not-a-list"}
        with pytest.raises(RuntimeError, match="all values must be lists of strings"):
            self._validate(checkpoint)

    def test_failed_symbols_list_elements_not_strings(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = {"2025-05-02": [123, 456]}
        with pytest.raises(RuntimeError, match="all values must be lists of strings"):
            self._validate(checkpoint)

    def test_failed_symbols_with_mixed_elements(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = {"2025-05-02": ["sh.600000", 123]}
        with pytest.raises(RuntimeError, match="all values must be lists of strings"):
            self._validate(checkpoint)

    def test_empty_failed_symbols_dict_is_valid(self):
        checkpoint = self._valid_base_checkpoint()
        checkpoint["failed_symbols_per_snapshot"] = {}
        self._validate(checkpoint)


# ---------------------------------------------------------------------------
# Test 7: replay end logs unresolved failure summary
# ---------------------------------------------------------------------------

class TestReplayEndUnresolvedSummary:
    """At the end of replay, if failed_symbols_per_snapshot has entries,
    a warning summary must be logged."""

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
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
    def test_end_logs_when_failures_exist(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results,
        mock_ensure, mock_exists, mock_getsize,
        mock_save_checkpoint, mock_load_checkpoint, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = _base_config()
        mock_stocks.return_value = ["sh.600000", "sh.600001", "sh.600002"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")

        # Make all symbols fail so both snapshots have failed symbols
        def scan_side_effect(symbol, df, context, **kwargs):
            raise ValueError("simulated error")
        strategy.scan.side_effect = scan_side_effect
        mock_create.return_value = strategy

        # Fresh run — no checkpoint
        mock_load_checkpoint.return_value = None

        def exists_side_effect(path):
            return False
        mock_exists.side_effect = exists_side_effect

        with caplog.at_level(logging.WARNING):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("scanner_app.VALIDATION_DIR", tmpdir):
                    run_weekly_replay_validation()

        # After replay completes, there should be an end-of-replay warning
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        unresolved_warnings = [r for r in warning_logs if "Unresolved failed symbols" in r.message]
        assert len(unresolved_warnings) == 1
        assert "2 snapshot(s)" in unresolved_warnings[0].message
        # 3 symbols * 2 snapshots = 6 total
        assert "6 total failed symbol" in unresolved_warnings[0].message

    @patch("scanner_app.load_replay_checkpoint")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.os.path.getsize", return_value=10)
    @patch("scanner_app.os.path.exists")
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
    def test_end_does_not_log_when_no_failures(
        self, mock_bs, mock_config, mock_stocks, mock_dates,
        mock_create, mock_load_full, mock_load_hist,
        mock_calc_returns, mock_write_errors, mock_write_results,
        mock_ensure, mock_exists, mock_getsize,
        mock_save_checkpoint, mock_load_checkpoint, caplog,
    ):
        mock_bs.login.return_value = MagicMock(error_code="0", error_msg="success")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = _base_config()
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        mock_load_checkpoint.return_value = None

        def exists_side_effect(path):
            return False
        mock_exists.side_effect = exists_side_effect

        with caplog.at_level(logging.WARNING):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("scanner_app.VALIDATION_DIR", tmpdir):
                    run_weekly_replay_validation()

        # No unresolved failure warnings at end
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        unresolved_warnings = [r for r in warning_logs if "Unresolved failed symbols" in r.message]
        assert len(unresolved_warnings) == 0


# ---------------------------------------------------------------------------
# Test: _summarize_failed_symbols helper
# ---------------------------------------------------------------------------

class TestSummarizeFailedSymbols:
    def test_empty_returns_empty_string(self):
        assert _summarize_failed_symbols({}) == ""

    def test_single_snapshot_single_symbol(self):
        result = _summarize_failed_symbols({"2025-05-02": ["sh.600000"]})
        assert "1 snapshot(s) with 1 total failed symbol(s)" in result
        assert "[2025-05-02] sh.600000" in result

    def test_multiple_snapshots_with_multiple_symbols(self):
        result = _summarize_failed_symbols({
            "2025-05-02": ["sh.600000", "sh.600001"],
            "2025-05-09": ["sh.600002", "sh.600003", "sh.600004"],
        })
        assert "2 snapshot(s) with 5 total failed symbol(s)" in result
        assert "[2025-05-02] sh.600000, sh.600001" in result
        assert "sh.600002, sh.600003, sh.600004" in result

    def test_truncates_more_than_5_snapshots(self):
        n = 8
        result = _summarize_failed_symbols({
            f"2025-05-{i+1:02d}": [f"sh.{600000 + i}"] for i in range(n)
        })
        assert "8 snapshot(s) with 8 total failed symbol(s)" in result
        assert "... and 3 more snapshot(s)" in result

    def test_truncates_more_than_3_symbols_per_snapshot(self):
        result = _summarize_failed_symbols({
            "2025-05-02": [f"sh.{600000 + i}" for i in range(5)],
        })
        assert "sh.600000, sh.600001, sh.600002, ... (5 total)" in result

    def test_exactly_5_snapshots_no_truncation(self):
        result = _summarize_failed_symbols({
            f"2025-05-{i+1:02d}": [f"sh.{600000 + i}"] for i in range(5)
        })
        assert "5 snapshot(s)" in result
        assert "... and" not in result
