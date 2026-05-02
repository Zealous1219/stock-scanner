"""Tests for replay_data_end_date fix: fixed data boundary for reproducibility.

Covers:
1. Fresh run computes and uses fixed replay_data_end_date
2. Resume reuses same replay_data_end_date from checkpoint
3. Old checkpoint missing replay_data_end_date rejects resume
4. replay_data_end_date mismatch rejects resume
5. Today changes but replay_data_end_date stays fixed
6. save_replay_checkpoint includes replay_data_end_date
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import pandas as pd

from scanner_app import (
    run_weekly_replay_validation,
    validate_replay_checkpoint,
    save_replay_checkpoint,
    load_full_df_for_replay,
)
from strategy_runtime import StrategyDecision, StrategyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(days: int = 120) -> pd.DataFrame:
    """Build a minimal DataFrame with `days` rows."""
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "open": [10.0] * days,
        "high": [10.5] * days,
        "low": [9.5] * days,
        "close": [10.0 + i * 0.01 for i in range(days)],
        "volume": [1000] * days,
    })


def _make_matched_result() -> StrategyResult:
    return StrategyResult(
        matched=True,
        reason_code="signal",
        reason_text="matched",
        details={"signal_date": "2025-06-01", "signal_type": "momentum_reversal"},
    )


def _make_bs_login_success():
    """Return a mock login result with success."""
    result = MagicMock()
    result.error_code = "0"
    result.error_msg = "success"
    return result


def _make_bs_logout_success():
    """Return a mock logout result with success."""
    result = MagicMock()
    result.error_code = "0"
    result.error_msg = "success"
    return result


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------

class TestReplayDataEndDate:
    """Verify replay_data_end_date fix for reproducible replays."""

    def _base_config(self):
        return {
            "strategy": {"params": {}},
            "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0},
        }

    # ------------------------------------------------------------------
    # Test 1: Fresh run uses fixed replay_data_end_date
    # ------------------------------------------------------------------

    def test_fresh_run_uses_fixed_replay_data_end_date(self):
        """Fresh run must compute replay_data_end_date from snapshot window
        and pass it to load_full_df_for_replay, not datetime.now()."""
        captured_end_dates = []

        def mock_load_full(symbol, *args, **kwargs):
            if 'replay_data_end_date' in kwargs:
                captured_end_dates.append(kwargs['replay_data_end_date'])
            return _make_df()

        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    last_snapshot = datetime(2025, 5, 16)
                    expected_end_date = (last_snapshot.date() + timedelta(weeks=21)).strftime("%Y-%m-%d")

                    with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                        datetime(2025, 5, 2, 23, 59, 59),
                        datetime(2025, 5, 9, 23, 59, 59),
                        datetime(2025, 5, 16, 23, 59, 59),
                    ]):
                        strategy = MagicMock()
                        strategy.name = "momentum_reversal_13"
                        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
                        strategy.scan.return_value = _make_matched_result()

                        with patch("scanner_app.create_strategy_from_config", return_value=strategy):
                            with patch("scanner_app.load_full_df_for_replay", side_effect=mock_load_full):
                                with patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df()):
                                    with patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15}):
                                        with patch("scanner_app.write_replay_results"):
                                            with patch("scanner_app.write_replay_errors"):
                                                with patch("scanner_app.os.path.exists", return_value=False):
                                                    with patch("scanner_app.os.path.getsize", return_value=0):
                                                        with patch("scanner_app.ensure_directories"):
                                                            with tempfile.TemporaryDirectory() as tmpdir:
                                                                with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                                    run_weekly_replay_validation()

        assert len(captured_end_dates) > 0, "load_full_df_for_replay should have been called with replay_data_end_date"
        assert captured_end_dates[0] == expected_end_date, (
            f"Expected replay_data_end_date={expected_end_date}, got {captured_end_dates[0]}"
        )

    # ------------------------------------------------------------------
    # Test 2: Resume reuses same replay_data_end_date
    # ------------------------------------------------------------------

    def test_resume_reuses_same_replay_data_end_date(self):
        """Resume must reuse replay_data_end_date from checkpoint, not recompute."""
        # Compute the expected end date: last snapshot + 21 weeks
        last_snapshot = datetime(2025, 5, 16)
        expected_end_date = (last_snapshot.date() + timedelta(weeks=21)).strftime("%Y-%m-%d")
        captured_end_dates = []

        def mock_load_full(symbol, *args, **kwargs):
            if 'replay_data_end_date' in kwargs:
                captured_end_dates.append(kwargs['replay_data_end_date'])
            return _make_df()

        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    checkpoint_data = {
                        "experiment_tag": "mr13_all_52w_v1",
                        "universe": "all",
                        "lookback_weeks": 52,
                        "version": "v1",
                        "snapshot_dates": ["2025-05-02", "2025-05-09", "2025-05-16"],
                        "completed_snapshots": ["2025-05-02", "2025-05-09"],
                        "replay_data_end_date": expected_end_date,  # Must match computed value
                    }

                    with patch("scanner_app.load_replay_checkpoint", return_value=checkpoint_data):
                        with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                            datetime(2025, 5, 2, 23, 59, 59),
                            datetime(2025, 5, 9, 23, 59, 59),
                            datetime(2025, 5, 16, 23, 59, 59),
                        ]):
                            strategy = MagicMock()
                            strategy.name = "momentum_reversal_13"
                            strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
                            strategy.scan.return_value = _make_matched_result()

                            with patch("scanner_app.create_strategy_from_config", return_value=strategy):
                                with patch("scanner_app.load_full_df_for_replay", side_effect=mock_load_full):
                                    with patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df()):
                                        with patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15}):
                                            with patch("scanner_app.write_replay_results"):
                                                with patch("scanner_app.write_replay_errors"):
                                                    with patch("scanner_app.os.path.exists") as mock_exists:
                                                        mock_exists.side_effect = lambda p: p.endswith(".checkpoint.json") or p.endswith(".csv")
                                                        with patch("scanner_app.os.path.getsize", return_value=10):
                                                                with patch("scanner_app.ensure_directories"):
                                                                    with tempfile.TemporaryDirectory() as tmpdir:
                                                                        with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                                            run_weekly_replay_validation()

        assert len(captured_end_dates) > 0, "load_full_df_for_replay should have been called"
        assert captured_end_dates[0] == expected_end_date, (
            f"Resume must reuse checkpoint's replay_data_end_date={expected_end_date}, "
            f"got {captured_end_dates[0]}"
        )

    # ------------------------------------------------------------------
    # Test 3: Old checkpoint missing replay_data_end_date rejects resume
    # ------------------------------------------------------------------

    def test_old_checkpoint_missing_replay_data_end_date_rejects_resume(self):
        """Old checkpoint without replay_data_end_date must be rejected on resume."""
        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    # Old checkpoint format: no replay_data_end_date
                    checkpoint_data = {
                        "experiment_tag": "mr13_all_52w_v1",
                        "universe": "all",
                        "lookback_weeks": 52,
                        "version": "v1",
                        "snapshot_dates": ["2025-05-02", "2025-05-09"],
                        "completed_snapshots": ["2025-05-02"],
                    }

                    with patch("scanner_app.load_replay_checkpoint", return_value=checkpoint_data):
                        with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                            datetime(2025, 5, 2, 23, 59, 59),
                            datetime(2025, 5, 9, 23, 59, 59),
                        ]):
                            with patch("scanner_app.os.path.exists") as mock_exists:
                                mock_exists.side_effect = lambda p: p.endswith(".checkpoint.json") or p.endswith(".csv")
                                with patch("scanner_app.ensure_directories"):
                                    with pytest.raises(RuntimeError) as exc_info:
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                run_weekly_replay_validation()

                                    message = str(exc_info.value)
                                    assert "replay_data_end_date" in message.lower(), (
                                        f"Error must mention replay_data_end_date: {message}"
                                    )
                                    assert "cannot safely resume" in message.lower(), (
                                        f"Error must mention cannot safely resume: {message}"
                                    )

    # ------------------------------------------------------------------
    # Test 4: replay_data_end_date mismatch rejects resume
    # ------------------------------------------------------------------

    def test_replay_data_end_date_mismatch_rejects_resume(self):
        """When checkpoint's replay_data_end_date differs from current run's
        computed value, resume must be rejected."""
        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    # Checkpoint has a different replay_data_end_date
                    checkpoint_data = {
                        "experiment_tag": "mr13_all_52w_v1",
                        "universe": "all",
                        "lookback_weeks": 52,
                        "version": "v1",
                        "snapshot_dates": ["2025-05-02", "2025-05-09"],
                        "completed_snapshots": ["2025-05-02"],
                        "replay_data_end_date": "2024-01-01",  # Wrong date
                    }

                    with patch("scanner_app.load_replay_checkpoint", return_value=checkpoint_data):
                        with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                            datetime(2025, 5, 2, 23, 59, 59),
                            datetime(2025, 5, 9, 23, 59, 59),
                        ]):
                            with patch("scanner_app.os.path.exists") as mock_exists:
                                mock_exists.side_effect = lambda p: p.endswith(".checkpoint.json") or p.endswith(".csv")
                                with patch("scanner_app.ensure_directories"):
                                    with pytest.raises(RuntimeError) as exc_info:
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                run_weekly_replay_validation()

                                    message = str(exc_info.value)
                                    assert "replay_data_end_date" in message.lower(), (
                                        f"Error must mention replay_data_end_date: {message}"
                                    )
                                    assert "does not match" in message.lower() or "mismatch" in message.lower(), (
                                        f"Error must mention mismatch: {message}"
                                    )

    # ------------------------------------------------------------------
    # Test 5: Today changes but replay_data_end_date stays fixed
    # ------------------------------------------------------------------

    def test_today_changes_but_replay_data_end_date_stays_fixed(self):
        """Even if datetime.now() returns different values on different days,
        replay_data_end_date must stay fixed for the same snapshot window."""
        captured_end_dates = []

        def mock_load_full(symbol, *args, **kwargs):
            if 'replay_data_end_date' in kwargs:
                captured_end_dates.append(kwargs['replay_data_end_date'])
            return _make_df()

        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    last_snapshot = datetime(2025, 5, 9)
                    expected_end_date = (last_snapshot.date() + timedelta(weeks=21)).strftime("%Y-%m-%d")

                    with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                        datetime(2025, 5, 2, 23, 59, 59),
                        datetime(2025, 5, 9, 23, 59, 59),
                    ]):
                        strategy = MagicMock()
                        strategy.name = "momentum_reversal_13"
                        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
                        strategy.scan.return_value = _make_matched_result()

                        with patch("scanner_app.create_strategy_from_config", return_value=strategy):
                            with patch("scanner_app.load_full_df_for_replay", side_effect=mock_load_full):
                                with patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df()):
                                    with patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15}):
                                        with patch("scanner_app.write_replay_results"):
                                            with patch("scanner_app.write_replay_errors"):
                                                with patch("scanner_app.os.path.exists", return_value=False):
                                                    with patch("scanner_app.os.path.getsize", return_value=0):
                                                            with patch("scanner_app.ensure_directories"):
                                                                # Simulate "first run"
                                                                with tempfile.TemporaryDirectory() as tmpdir:
                                                                    with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                                        run_weekly_replay_validation()

                                                                # Simulate "second run" on a different day
                                                                with tempfile.TemporaryDirectory() as tmpdir2:
                                                                    with patch("scanner_app.VALIDATION_DIR", tmpdir2):
                                                                        run_weekly_replay_validation()

        assert len(captured_end_dates) >= 2, (
            f"Expected at least 2 calls to load_full_df_for_replay, got {len(captured_end_dates)}"
        )
        assert captured_end_dates[0] == expected_end_date, (
            f"First run: expected {expected_end_date}, got {captured_end_dates[0]}"
        )
        assert captured_end_dates[0] == captured_end_dates[-1], (
            f"replay_data_end_date must be identical across runs: "
            f"{captured_end_dates[0]} != {captured_end_dates[-1]}"
        )

    # ------------------------------------------------------------------
    # Test 6: save_replay_checkpoint includes replay_data_end_date
    # ------------------------------------------------------------------

    def test_save_checkpoint_includes_replay_data_end_date(self):
        """save_replay_checkpoint must be called with replay_data_end_date."""
        saved_checkpoints = []

        def mock_save_checkpoint(*args, **kwargs):
            saved_checkpoints.append(kwargs)

        with patch("scanner_app.bs") as mock_bs:
            mock_bs.login.return_value = _make_bs_login_success()
            mock_bs.logout.return_value = _make_bs_logout_success()

            with patch("scanner_app.load_config", return_value=self._base_config()):
                with patch("scanner_app.get_stock_list", return_value=["sh.600000"]):
                    last_snapshot = datetime(2025, 5, 9)
                    expected_end_date = (last_snapshot.date() + timedelta(weeks=21)).strftime("%Y-%m-%d")

                    with patch("scanner_app.generate_weekly_snapshot_dates", return_value=[
                        datetime(2025, 5, 2, 23, 59, 59),
                        datetime(2025, 5, 9, 23, 59, 59),
                    ]):
                        strategy = MagicMock()
                        strategy.name = "momentum_reversal_13"
                        strategy.can_run.return_value = StrategyDecision(should_run=True, reason_code="ok", reason_text="ok")
                        strategy.scan.return_value = _make_matched_result()

                        with patch("scanner_app.create_strategy_from_config", return_value=strategy):
                            with patch("scanner_app.load_full_df_for_replay", return_value=_make_df()):
                                with patch("scanner_app.load_historical_data_up_to_date", return_value=_make_df()):
                                    with patch("scanner_app.calculate_forward_returns", return_value={"return_4w": 0.05, "return_8w": 0.1, "return_12w": 0.15}):
                                        with patch("scanner_app.write_replay_results"):
                                            with patch("scanner_app.write_replay_errors"):
                                                with patch("scanner_app.save_replay_checkpoint", side_effect=mock_save_checkpoint):
                                                    with patch("scanner_app.os.path.exists", return_value=False):
                                                        with patch("scanner_app.os.path.getsize", return_value=0):
                                                                with patch("scanner_app.ensure_directories"):
                                                                    with tempfile.TemporaryDirectory() as tmpdir:
                                                                        with patch("scanner_app.VALIDATION_DIR", tmpdir):
                                                                            run_weekly_replay_validation()

        assert len(saved_checkpoints) > 0, "save_replay_checkpoint should have been called"
        found = False
        for kwargs in saved_checkpoints:
            if 'replay_data_end_date' in kwargs:
                assert kwargs['replay_data_end_date'] == expected_end_date, (
                    f"Expected replay_data_end_date={expected_end_date} in checkpoint, "
                    f"got {kwargs['replay_data_end_date']}"
                )
                found = True
        assert found, "replay_data_end_date not found in save_replay_checkpoint kwargs"
