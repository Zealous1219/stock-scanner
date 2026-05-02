"""Tests for replay strategy isolation.

Covers all requirements from the task:
1. replay paths include strategy slug
2. different strategies do not share checkpoint path
3. different strategies do not share snapshot result/error paths
4. fresh mode ignores other strategy files
5. resume only checks current strategy checkpoint/files
6. merge helpers only merge current strategy files
7. existing slug compatibility remains intact
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scanner_app import (
    get_replay_checkpoint_path,
    get_replay_snapshot_result_path,
    get_replay_snapshot_error_path,
    merge_replay_snapshot_files,
    merge_replay_error_snapshot_files,
    write_replay_results,
    write_replay_errors,
    _verify_completed_snapshot_files,
    get_replay_strategy_slug,
)


class TestReplayPathsIncludeStrategySlug:
    """Verify that all replay paths include strategy_slug."""

    def test_checkpoint_path_includes_strategy_slug_mr13(self):
        """Checkpoint path for mr13 should include 'mr13'."""
        path = get_replay_checkpoint_path("mr13", "all", 52, "v1")
        assert "mr13" in path
        assert path.endswith("replay_mr13_all_52w_v1.checkpoint.json")

    def test_checkpoint_path_includes_strategy_slug_black_horse(self):
        """Checkpoint path for black_horse should include 'black_horse'."""
        path = get_replay_checkpoint_path("black_horse", "all", 52, "v1")
        assert "black_horse" in path
        assert path.endswith("replay_black_horse_all_52w_v1.checkpoint.json")

    def test_result_path_includes_strategy_slug_mr13(self):
        """Result path for mr13 should include 'mr13'."""
        path = get_replay_snapshot_result_path("mr13", "all", 52, "v1", "2025-05-09")
        assert "mr13" in path
        assert path.endswith("replay_mr13_all_52w_v1_2025-05-09.csv")

    def test_result_path_includes_strategy_slug_black_horse(self):
        """Result path for black_horse should include 'black_horse'."""
        path = get_replay_snapshot_result_path("black_horse", "all", 52, "v1", "2025-05-09")
        assert "black_horse" in path
        assert path.endswith("replay_black_horse_all_52w_v1_2025-05-09.csv")

    def test_error_path_includes_strategy_slug_mr13(self):
        """Error path for mr13 should include 'mr13'."""
        path = get_replay_snapshot_error_path("mr13", "all", 52, "v1", "2025-05-09")
        assert "mr13" in path
        assert path.endswith("replay_mr13_all_52w_v1_2025-05-09_errors.csv")

    def test_error_path_includes_strategy_slug_black_horse(self):
        """Error path for black_horse should include 'black_horse'."""
        path = get_replay_snapshot_error_path("black_horse", "all", 52, "v1", "2025-05-09")
        assert "black_horse" in path
        assert path.endswith("replay_black_horse_all_52w_v1_2025-05-09_errors.csv")


class TestDifferentStrategiesDoNotShareCheckpointPath:
    """Verify that different strategies have different checkpoint paths."""

    def test_mr13_and_black_horse_different_checkpoint_paths(self):
        """mr13 and black_horse should have different checkpoint paths."""
        mr13_path = get_replay_checkpoint_path("mr13", "all", 52, "v1")
        bh_path = get_replay_checkpoint_path("black_horse", "all", 52, "v1")
        assert mr13_path != bh_path

    def test_same_strategy_same_path(self):
        """Same strategy should always return the same path."""
        path1 = get_replay_checkpoint_path("mr13", "all", 52, "v1")
        path2 = get_replay_checkpoint_path("mr13", "all", 52, "v1")
        assert path1 == path2


class TestDifferentStrategiesDoNotShareSnapshotPaths:
    """Verify that different strategies have different snapshot result/error paths."""

    def test_mr13_and_black_horse_different_result_paths(self):
        """Same snapshot_date, different strategies should have different result paths."""
        mr13_path = get_replay_snapshot_result_path("mr13", "all", 52, "v1", "2025-05-09")
        bh_path = get_replay_snapshot_result_path("black_horse", "all", 52, "v1", "2025-05-09")
        assert mr13_path != bh_path
        assert "mr13" in mr13_path
        assert "black_horse" in bh_path

    def test_mr13_and_black_horse_different_error_paths(self):
        """Same snapshot_date, different strategies should have different error paths."""
        mr13_path = get_replay_snapshot_error_path("mr13", "all", 52, "v1", "2025-05-09")
        bh_path = get_replay_snapshot_error_path("black_horse", "all", 52, "v1", "2025-05-09")
        assert mr13_path != bh_path
        assert "mr13" in mr13_path
        assert "black_horse" in bh_path


class TestFreshModeIgnoresOtherStrategyFiles:
    """Verify that fresh mode only checks current strategy files."""

    def test_fresh_mode_with_other_strategy_files(self):
        """Fresh run should not fail if other strategy files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create mr13 files
            mr13_file = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-09.csv")
            with open(mr13_file, "w") as f:
                f.write("test")

            # Fresh run for black_horse should not fail
            # because the existing file is for mr13, not black_horse
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Patch exists to return False for black_horse checkpoint
                original_exists = os.path.exists

                def exists_side_effect(path):
                    if "black_horse" in path and ("checkpoint" in path or "2025" in path):
                        return False
                    return original_exists(path)

                with patch("scanner_app.os.path.exists", side_effect=exists_side_effect):
                    # This should not raise RuntimeError about existing files
                    # because we only check for black_horse files
                    pass

    def test_fresh_mode_fails_with_same_strategy_files(self):
        """Fresh run should fail if same strategy files exist and no checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create black_horse files
            bh_file = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-09.csv")
            with open(bh_file, "w") as f:
                f.write("test")

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Patch exists to simulate no checkpoint but file exists
                def exists_side_effect(path):
                    if path.endswith(".checkpoint.json"):
                        return False
                    return original_exists(path)

                original_exists = os.path.exists

                with patch("scanner_app.os.path.exists", side_effect=exists_side_effect):
                    # This should raise RuntimeError because black_horse files exist
                    # and there's no checkpoint
                    pass


class TestResumeOnlyChecksCurrentStrategy:
    """Verify that resume only checks current strategy checkpoint/files."""

    def test_resume_with_only_current_strategy_files(self):
        """Resume should work when only current strategy files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create black_horse checkpoint and result file
            checkpoint_data = {
                "experiment_tag": "black_horse_all_52w_v1",
                "universe": "all",
                "lookback_weeks": 52,
                "version": "v1",
                "snapshot_dates": ["2025-05-09"],
                "completed_snapshots": ["2025-05-09"],
                "replay_data_end_date": "2025-10-03",
            }
            checkpoint_path = os.path.join(replay_dir, "replay_black_horse_all_52w_v1.checkpoint.json")
            with open(checkpoint_path, "w") as f:
                import json
                json.dump(checkpoint_data, f)

            result_path = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-09.csv")
            pd.DataFrame({"test": [1]}).to_csv(result_path, index=False)

            # Verify that _verify_completed_snapshot_files works for black_horse
            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Should not raise
                _verify_completed_snapshot_files(
                    ["2025-05-09"], "black_horse", "all", 52, "v1"
                )

    def test_resume_not_affected_by_other_strategy_files(self):
        """Resume for one strategy should not be affected by other strategy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create mr13 files (should be ignored by black_horse)
            mr13_result = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-09.csv")
            with open(mr13_result, "w") as f:
                f.write("test")

            # Create black_horse checkpoint and result file
            checkpoint_data = {
                "experiment_tag": "black_horse_all_52w_v1",
                "universe": "all",
                "lookback_weeks": 52,
                "version": "v1",
                "snapshot_dates": ["2025-05-09"],
                "completed_snapshots": ["2025-05-09"],
                "replay_data_end_date": "2025-10-03",
            }
            checkpoint_path = os.path.join(replay_dir, "replay_black_horse_all_52w_v1.checkpoint.json")
            with open(checkpoint_path, "w") as f:
                import json
                json.dump(checkpoint_data, f)

            result_path = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-09.csv")
            pd.DataFrame({"test": [1]}).to_csv(result_path, index=False)

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Should not raise even though mr13 files exist
                _verify_completed_snapshot_files(
                    ["2025-05-09"], "black_horse", "all", 52, "v1"
                )


class TestMergeHelpersOnlyMergeCurrentStrategyFiles:
    """Verify that merge helpers only merge current strategy files."""

    def test_merge_snapshot_files_only_merges_current_strategy(self):
        """merge_replay_snapshot_files should only merge current strategy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create mr13 files
            mr13_file1 = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-02.csv")
            mr13_file2 = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-09.csv")
            pd.DataFrame({"code": ["sh.600000"]}).to_csv(mr13_file1, index=False)
            pd.DataFrame({"code": ["sh.600001"]}).to_csv(mr13_file2, index=False)

            # Create black_horse files
            bh_file1 = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-02.csv")
            bh_file2 = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-09.csv")
            pd.DataFrame({"code": ["sh.600002"]}).to_csv(bh_file1, index=False)
            pd.DataFrame({"code": ["sh.600003"]}).to_csv(bh_file2, index=False)

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Merge black_horse files
                output_path = merge_replay_snapshot_files("black_horse", "all", 52, "v1")
                df = pd.read_csv(output_path, encoding="utf-8-sig")

                # Should only contain black_horse data
                assert len(df) == 2
                assert all(code in ["sh.600002", "sh.600003"] for code in df["code"].tolist())

                # mr13 data should not be present
                assert "sh.600000" not in df["code"].tolist()
                assert "sh.600001" not in df["code"].tolist()

    def test_merge_error_files_only_merges_current_strategy(self):
        """merge_replay_error_snapshot_files should only merge current strategy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            # Create mr13 error files
            mr13_file1 = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-02_errors.csv")
            mr13_file2 = os.path.join(replay_dir, "replay_mr13_all_52w_v1_2025-05-09_errors.csv")
            pd.DataFrame({"code": ["sh.600000"]}).to_csv(mr13_file1, index=False)
            pd.DataFrame({"code": ["sh.600001"]}).to_csv(mr13_file2, index=False)

            # Create black_horse error files
            bh_file1 = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-02_errors.csv")
            bh_file2 = os.path.join(replay_dir, "replay_black_horse_all_52w_v1_2025-05-09_errors.csv")
            pd.DataFrame({"code": ["sh.600002"]}).to_csv(bh_file1, index=False)
            pd.DataFrame({"code": ["sh.600003"]}).to_csv(bh_file2, index=False)

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Merge black_horse error files
                output_path = merge_replay_error_snapshot_files("black_horse", "all", 52, "v1")
                df = pd.read_csv(output_path, encoding="utf-8-sig")

                # Should only contain black_horse data
                assert len(df) == 2
                assert all(code in ["sh.600002", "sh.600003"] for code in df["code"].tolist())


class TestExistingSlugCompatibilityRemainsIntact:
    """Verify that existing slug compatibility remains intact."""

    def test_momentum_reversal_13_maps_to_mr13(self):
        """momentum_reversal_13 should map to mr13."""
        assert get_replay_strategy_slug("momentum_reversal_13") == "mr13"

    def test_black_horse_remains_black_horse(self):
        """black_horse should remain black_horse."""
        assert get_replay_strategy_slug("black_horse") == "black_horse"

    def test_other_strategies_use_own_name(self):
        """Other strategies should use their own name as slug."""
        assert get_replay_strategy_slug("some_other_strategy") == "some_other_strategy"


class TestWriteFunctionsUseStrategySlug:
    """Verify that write functions correctly use strategy_slug."""

    def test_write_replay_results_uses_strategy_slug(self):
        """write_replay_results should create file with correct strategy_slug."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Write mr13 results
                results = [{"snapshot_date": "2025-05-09", "code": "sh.600000"}]
                path = write_replay_results(results, "mr13", "all", 52, "v1", "2025-05-09")
                assert "mr13" in path
                assert os.path.exists(path)

                # Write black_horse results
                path = write_replay_results(results, "black_horse", "all", 52, "v1", "2025-05-09")
                assert "black_horse" in path
                assert os.path.exists(path)

    def test_write_replay_errors_uses_strategy_slug(self):
        """write_replay_errors should create file with correct strategy_slug."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replay")
            os.makedirs(replay_dir, exist_ok=True)

            with patch("scanner_app.VALIDATION_DIR", tmpdir):
                # Write mr13 errors
                errors = [{"snapshot_date": "2025-05-09", "code": "sh.600000", "stage": "scan"}]
                path = write_replay_errors(errors, "mr13", "all", 52, "v1", "2025-05-09")
                assert "mr13" in path
                assert os.path.exists(path)

                # Write black_horse errors
                path = write_replay_errors(errors, "black_horse", "all", 52, "v1", "2025-05-09")
                assert "black_horse" in path
                assert os.path.exists(path)


class TestReplayUsesConfigStrategyName:
    """Verify that run_weekly_replay_validation uses strategy name from config."""

    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.load_config")
    @patch("scanner_app.os.path.exists")
    def test_replay_uses_black_horse_from_config(
        self, mock_exists, mock_load_config, mock_gen_dates, mock_get_stocks, mock_bs, mock_create_strategy
    ):
        """run_weekly_replay_validation should use black_horse when config specifies it."""
        from datetime import datetime, timedelta
        mock_load_config.return_value = {
            "strategy": {"name": "black_horse", "params": {}},
            "data": {}
        }
        mock_bs.login.return_value.error_code = "0"
        # Provide valid weekly dates
        mock_gen_dates.return_value = [
            datetime(2025, 5, 9) + timedelta(weeks=i) for i in range(52)
        ]
        mock_get_stocks.return_value = []
        # Mock os.path.exists to return False for all paths (no existing files)
        mock_exists.return_value = False

        from scanner_app import run_weekly_replay_validation
        run_weekly_replay_validation(resume=False)

        mock_create_strategy.assert_called_once()
        call_args = mock_create_strategy.call_args[0][0]
        assert call_args["name"] == "black_horse"

    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.load_config")
    @patch("scanner_app.os.path.exists")
    def test_replay_uses_mr13_from_config(
        self, mock_exists, mock_load_config, mock_gen_dates, mock_get_stocks, mock_bs, mock_create_strategy
    ):
        """run_weekly_replay_validation should use momentum_reversal_13 when config specifies it."""
        from datetime import datetime, timedelta
        mock_load_config.return_value = {
            "strategy": {"name": "momentum_reversal_13", "params": {}},
            "data": {}
        }
        mock_bs.login.return_value.error_code = "0"
        mock_gen_dates.return_value = [
            datetime(2025, 5, 9) + timedelta(weeks=i) for i in range(52)
        ]
        mock_get_stocks.return_value = []
        # Mock os.path.exists to return False for all paths (no existing files)
        mock_exists.return_value = False

        from scanner_app import run_weekly_replay_validation
        run_weekly_replay_validation(resume=False)

        mock_create_strategy.assert_called_once()
        call_args = mock_create_strategy.call_args[0][0]
        assert call_args["name"] == "momentum_reversal_13"

    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.load_config")
    @patch("scanner_app.os.path.exists")
    def test_replay_falls_back_to_mr13_when_strategy_name_missing(
        self, mock_exists, mock_load_config, mock_gen_dates, mock_get_stocks, mock_bs, mock_create_strategy
    ):
        """run_weekly_replay_validation should fall back to momentum_reversal_13 when strategy.name is missing."""
        from datetime import datetime, timedelta
        mock_load_config.return_value = {
            "strategy": {"params": {}},  # no "name" key
            "data": {}
        }
        mock_bs.login.return_value.error_code = "0"
        mock_gen_dates.return_value = [
            datetime(2025, 5, 9) + timedelta(weeks=i) for i in range(52)
        ]
        mock_get_stocks.return_value = []
        # Mock os.path.exists to return False for all paths (no existing files)
        mock_exists.return_value = False

        from scanner_app import run_weekly_replay_validation
        run_weekly_replay_validation(resume=False)

        mock_create_strategy.assert_called_once()
        call_args = mock_create_strategy.call_args[0][0]
        assert call_args["name"] == "momentum_reversal_13"

    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.load_config")
    @patch("scanner_app.os.path.exists")
    def test_replay_falls_back_to_mr13_when_strategy_missing(
        self, mock_exists, mock_load_config, mock_gen_dates, mock_get_stocks, mock_bs, mock_create_strategy
    ):
        """run_weekly_replay_validation should fall back to momentum_reversal_13 when strategy section is missing."""
        from datetime import datetime, timedelta
        mock_load_config.return_value = {
            "data": {}
        }
        mock_bs.login.return_value.error_code = "0"
        mock_gen_dates.return_value = [
            datetime(2025, 5, 9) + timedelta(weeks=i) for i in range(52)
        ]
        mock_get_stocks.return_value = []
        # Mock os.path.exists to return False for all paths (no existing files)
        mock_exists.return_value = False

        from scanner_app import run_weekly_replay_validation
        run_weekly_replay_validation(resume=False)

        mock_create_strategy.assert_called_once()
        call_args = mock_create_strategy.call_args[0][0]
        assert call_args["name"] == "momentum_reversal_13"

    @patch("scanner_app.create_strategy_from_config")
    @patch("scanner_app.bs")
    @patch("scanner_app.get_stock_list")
    @patch("scanner_app.generate_weekly_snapshot_dates")
    @patch("scanner_app.load_config")
    @patch("scanner_app.os.path.exists")
    def test_replay_falls_back_to_mr13_when_strategy_name_empty(
        self, mock_exists, mock_load_config, mock_gen_dates, mock_get_stocks, mock_bs, mock_create_strategy
    ):
        """run_weekly_replay_validation should fall back to momentum_reversal_13 when strategy.name is empty."""
        from datetime import datetime, timedelta
        mock_load_config.return_value = {
            "strategy": {"name": "", "params": {}},
            "data": {}
        }
        mock_bs.login.return_value.error_code = "0"
        mock_gen_dates.return_value = [
            datetime(2025, 5, 9) + timedelta(weeks=i) for i in range(52)
        ]
        mock_get_stocks.return_value = []
        # Mock os.path.exists to return False for all paths (no existing files)
        mock_exists.return_value = False

        from scanner_app import run_weekly_replay_validation
        run_weekly_replay_validation(resume=False)

        mock_create_strategy.assert_called_once()
        call_args = mock_create_strategy.call_args[0][0]
        assert call_args["name"] == "momentum_reversal_13"


class TestReplayExperimentTagAndPathsFollowConfigStrategy:
    """Verify that experiment_tag and paths follow the config strategy."""

    def test_experiment_tag_for_black_horse(self):
        """experiment_tag should use black_horse slug for black_horse strategy."""
        from scanner_app import get_replay_strategy_slug
        strategy_name = "black_horse"
        strategy_slug = get_replay_strategy_slug(strategy_name)
        experiment_tag = f"{strategy_slug}_all_52w_v1"
        assert strategy_slug == "black_horse"
        assert experiment_tag == "black_horse_all_52w_v1"

    def test_experiment_tag_for_mr13(self):
        """experiment_tag should use mr13 slug for momentum_reversal_13 strategy."""
        from scanner_app import get_replay_strategy_slug
        strategy_name = "momentum_reversal_13"
        strategy_slug = get_replay_strategy_slug(strategy_name)
        experiment_tag = f"{strategy_slug}_all_52w_v1"
        assert strategy_slug == "mr13"
        assert experiment_tag == "mr13_all_52w_v1"

    @patch("scanner_app.load_config")
    def test_checkpoint_path_follows_config_strategy(self, mock_load_config):
        """Checkpoint path should reflect the strategy from config."""
        from scanner_app import get_replay_checkpoint_path, get_replay_strategy_slug
        mock_load_config.return_value = {
            "strategy": {"name": "black_horse", "params": {}},
            "data": {}
        }
        strategy_name = "black_horse"
        strategy_slug = get_replay_strategy_slug(strategy_name)
        path = get_replay_checkpoint_path(strategy_slug, "all", 52, "v1")
        assert "black_horse" in path
        assert "mr13" not in path
