"""Tests for per-snapshot replay output.

Covers:
- Per-snapshot result file naming
- Per-snapshot error file naming
- Overwrite semantics (no append)
- Checkpoint updated only after file write
- Resume skips completed snapshots without requiring aggregate CSV
- Inconsistent state detection
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest

from scanner_app import (
    write_replay_results,
    write_replay_errors,
    get_replay_snapshot_result_path,
    get_replay_snapshot_error_path,
    run_weekly_replay_validation,
    load_replay_checkpoint,
    save_replay_checkpoint,
    get_replay_checkpoint_path,
    _verify_completed_snapshot_files,
)
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


# ---------------------------------------------------------------------------
# Test: per-snapshot file naming
# ---------------------------------------------------------------------------

class TestPerSnapshotFileNaming:
    """Verify that per-snapshot files are named correctly."""

    def test_result_file_path_includes_snapshot_date(self):
        """get_replay_snapshot_result_path should include snapshot_date in filename."""
        path = get_replay_snapshot_result_path("all", 52, "v1", "2025-05-09")
        assert "2025-05-09" in path
        assert path.endswith("replay_all_52w_v1_2025-05-09.csv")

    def test_error_file_path_includes_snapshot_date(self):
        """get_replay_snapshot_error_path should include snapshot_date in filename."""
        path = get_replay_snapshot_error_path("all", 52, "v1", "2025-05-09")
        assert "2025-05-09" in path
        assert path.endswith("replay_all_52w_v1_2025-05-09_errors.csv")

    def test_write_replay_results_creates_file_with_snapshot_date(self):
        """write_replay_results should create file with snapshot_date in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create replay subdirectory
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)
            # Patch VALIDATION_DIR to use temp directory
            result = [{"snapshot_date": "2025-05-09", "code": "sh.600000"}]
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                path = write_replay_results(result, "all", 52, "v1", "2025-05-09")
                assert os.path.exists(path)
                assert "2025-05-09" in path

    def test_write_replay_errors_creates_file_with_snapshot_date(self):
        """write_replay_errors should create file with snapshot_date in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create replay subdirectory
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)
            errors = [{"snapshot_date": "2025-05-09", "code": "sh.600000", "stage": "scan"}]
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                path = write_replay_errors(errors, "all", 52, "v1", "2025-05-09")
                assert os.path.exists(path)
                assert "2025-05-09" in path


# ---------------------------------------------------------------------------
# Test: overwrite semantics (no append)
# ---------------------------------------------------------------------------

class TestOverwriteSemantics:
    """Verify that re-running same snapshot overwrites, not appends."""

    def _ensure_replay_dir(self, tmpdir):
        """Create replay subdirectory in tmpdir."""
        replay_dir = os.path.join(tmpdir, "replay")
        os.makedirs(replay_dir, exist_ok=True)
        return replay_dir

    def test_rerun_same_snapshot_overwrites_results(self):
        """Running same snapshot twice should overwrite, not append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ensure_replay_dir(tmpdir)
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # First write
                results1 = [
                    {"snapshot_date": "2025-05-09", "code": "sh.600000", "value": 1},
                ]
                write_replay_results(results1, "all", 52, "v1", "2025-05-09")

                # Second write with different data
                results2 = [
                    {"snapshot_date": "2025-05-09", "code": "sh.600001", "value": 2},
                ]
                write_replay_results(results2, "all", 52, "v1", "2025-05-09")

                # Read back and verify only second write exists
                path = get_replay_snapshot_result_path("all", 52, "v1", "2025-05-09")
                df = pd.read_csv(path, encoding="utf-8-sig")
                assert len(df) == 1
                assert df.iloc[0]["code"] == "sh.600001"

    def test_rerun_same_snapshot_overwrites_errors(self):
        """Running same snapshot twice should overwrite error file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ensure_replay_dir(tmpdir)
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # First write
                errors1 = [
                    {"snapshot_date": "2025-05-09", "code": "sh.600000", "stage": "scan"},
                ]
                write_replay_errors(errors1, "all", 52, "v1", "2025-05-09")

                # Second write with different data
                errors2 = [
                    {"snapshot_date": "2025-05-09", "code": "sh.600001", "stage": "load"},
                ]
                write_replay_errors(errors2, "all", 52, "v1", "2025-05-09")

                # Read back and verify only second write exists
                path = get_replay_snapshot_error_path("all", 52, "v1", "2025-05-09")
                df = pd.read_csv(path, encoding="utf-8-sig")
                assert len(df) == 1
                assert df.iloc[0]["code"] == "sh.600001"

    def test_no_duplicate_rows_on_rerun(self):
        """Ensure no duplicate rows when re-running same snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ensure_replay_dir(tmpdir)
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                results = [
                    {"snapshot_date": "2025-05-09", "code": "sh.600000"},
                ]

                # Write 3 times
                for _ in range(3):
                    write_replay_results(results, "all", 52, "v1", "2025-05-09")

                path = get_replay_snapshot_result_path("all", 52, "v1", "2025-05-09")
                df = pd.read_csv(path, encoding="utf-8-sig")
                # Should still be only 1 row, not 3
                assert len(df) == 1


# ---------------------------------------------------------------------------
# Test: checkpoint updated only after file write
# ---------------------------------------------------------------------------

class TestCheckpointUpdateTiming:
    """Verify checkpoint is updated only after snapshot files are written."""

    @patch("scanner_app.bs")
    @patch("scanner_app.load_config")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05})
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.os.path.exists")
    def test_checkpoint_saved_after_file_write(
        self, mock_exists, mock_ensure, mock_save_cp,
        mock_calc, mock_load_hist, mock_load_full, mock_create,
        mock_dates, mock_stocks, mock_config, mock_bs,
    ):
        """Checkpoint should be saved after write_replay_results is called."""
        mock_bs.login.return_value = MagicMock(error_code="0")
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

        # Make exists return False (fresh run, no checkpoint)
        mock_exists.return_value = False

        # Track call order
        call_order = []

        def track_write_results(*args, **kwargs):
            call_order.append("write_results")
            return "result.csv"

        def track_save_checkpoint(*args, **kwargs):
            call_order.append("save_checkpoint")

        mock_save_cp.side_effect = track_save_checkpoint

        with patch("scanner_app.write_replay_results", side_effect=track_write_results):
            with patch("scanner_app.write_replay_errors"):
                run_weekly_replay_validation()

        # Verify write_replay_results was called before save_replay_checkpoint
        assert "write_results" in call_order
        assert "save_checkpoint" in call_order
        assert call_order.index("write_results") < call_order.index("save_checkpoint")


# ---------------------------------------------------------------------------
# Test: resume skips completed snapshots
# ---------------------------------------------------------------------------

class TestResumeSkipsCompleted:
    """Verify resume skips completed snapshots and doesn't require aggregate CSV."""

    @patch("scanner_app.bs")
    @patch("scanner_app.load_config")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05})
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.save_replay_checkpoint")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.os.path.exists")
    def test_resume_skips_completed_without_aggregate_csv(
        self, mock_exists, mock_ensure, mock_save_cp,
        mock_write_errors, mock_write_results, mock_calc,
        mock_load_hist, mock_load_full, mock_create,
        mock_dates, mock_stocks, mock_config, mock_bs,
    ):
        """Resume should skip completed snapshots without requiring aggregate replay.csv."""
        mock_bs.login.return_value = MagicMock(error_code="0")
        mock_bs.logout.return_value = MagicMock()
        mock_config.return_value = {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }
        mock_stocks.return_value = ["sh.600000"]
        mock_dates.return_value = [datetime(2025, 5, 2), datetime(2025, 5, 9)]

        strategy = MagicMock()
        strategy.name = "momentum_reversal_13"
        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
        strategy.scan.return_value = _make_matched_result()
        mock_create.return_value = strategy

        # Simulate checkpoint exists with one completed snapshot
        # Importantly, we return False for replay.csv (aggregate file should NOT be required)
        def exists_side_effect(path):
            if path.endswith(".checkpoint.json"):
                return True
            if "2025-05-02" in path:
                # Completed snapshot result file exists
                return True
            if path.endswith(".csv"):
                # Other CSV files (aggregate) should NOT be required
                return False
            return False

        mock_exists.side_effect = exists_side_effect

        # Mock load_replay_checkpoint to return checkpoint with completed snapshot
        checkpoint_data = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-02", "2025-05-09"],
            "completed_snapshots": ["2025-05-02"],
            "replay_data_end_date": "2025-10-03",
        }

        with patch("scanner_app.load_replay_checkpoint", return_value=checkpoint_data):
            run_weekly_replay_validation()

        # Should have written results only for 2025-05-09 (not 2025-05-02)
        write_calls = mock_write_results.call_args_list
        assert len(write_calls) == 1
        # The snapshot_date in the results should be 2025-05-09
        results_arg = write_calls[0].args[0]
        assert results_arg[0]["snapshot_date"] == "2025-05-09"


# ---------------------------------------------------------------------------
# Test: inconsistent state detection
# ---------------------------------------------------------------------------

class TestInconsistentStateDetection:
    """Verify that inconsistent state is detected and reported."""

    def test_completed_snapshot_missing_result_file_raises_error(self):
        """If checkpoint marks snapshot as completed but result file is missing, raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                completed_snapshots = ["2025-05-09"]

                # Don't create the result file
                with pytest.raises(RuntimeError, match="inconsistent state"):
                    _verify_completed_snapshot_files(completed_snapshots, "all", 52, "v1")

    def test_all_completed_snapshots_have_files_passes(self):
        """If all completed snapshots have result files, no error is raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create replay subdirectory
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                completed_snapshots = ["2025-05-09"]

                # Create the result file
                path = get_replay_snapshot_result_path("all", 52, "v1", "2025-05-09")
                pd.DataFrame({"test": [1]}).to_csv(path, index=False)

                # Should not raise
                _verify_completed_snapshot_files(completed_snapshots, "all", 52, "v1")

    @patch("scanner_app.bs")
    @patch("scanner_app.load_config")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.load_full_df_for_replay", return_value=_make_df())
    @patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df())
    @patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05})
    @patch("scanner_app.write_replay_results")
    @patch("scanner_app.write_replay_errors")
    @patch("scanner_app.ensure_directories")
    @patch("scanner_app.os.path.exists")
    def test_resume_with_missing_result_file_raises_error(
        self, mock_exists, mock_ensure,
        mock_write_errors, mock_write_results,
        mock_calc, mock_load_hist, mock_load_full, mock_create,
        mock_dates, mock_stocks, mock_config, mock_bs,
    ):
        """Resume should fail if checkpoint exists but result file is missing."""
        mock_bs.login.return_value = MagicMock(error_code="0")
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

        # Simulate checkpoint exists
        def exists_side_effect(path):
            if path.endswith(".checkpoint.json"):
                return True
            # Result file doesn't exist
            if path.endswith(".csv"):
                return False
            return False

        mock_exists.side_effect = exists_side_effect

        checkpoint_data = {
            "experiment_tag": "mr13_all_52w_v1",
            "universe": "all",
            "lookback_weeks": 52,
            "version": "v1",
            "snapshot_dates": ["2025-05-09"],
            "completed_snapshots": ["2025-05-09"],
            "replay_data_end_date": "2025-10-03",
        }

        with patch("scanner_app.load_replay_checkpoint", return_value=checkpoint_data):
            with pytest.raises(RuntimeError, match="result files are missing"):
                run_weekly_replay_validation()
