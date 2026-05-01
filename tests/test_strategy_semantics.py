"""Real strategy semantic tests for mr13 and black_horse.

These tests verify core business logic by constructing realistic weekly bar
examples and feeding them directly to strategy methods.  They do NOT rely
on the replay/mock framework.

Each test is designed to be readable as a self-contained business-rule
specification — the data layout makes it obvious *which* rule is being
exercised.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from strategies.momentum_reversal_13 import MomentumReversal13Strategy
from strategies.black_horse import BlackHorseStrategy
from strategy_runtime import StrategyContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> StrategyContext:
    return StrategyContext(now=datetime(2025, 12, 31), stock_pool="test")


def _make_weekly(closes: list[float],
                 volumes: list[int] | None = None) -> pd.DataFrame:
    """Build a weekly DataFrame from close prices.

    OHLC is derived from close for simplicity:
      open = close * 0.99, high = close * 1.02, low = close * 0.98
    """
    n = len(closes)
    dates = pd.date_range("2024-01-05", periods=n, freq="W-FRI")
    if volumes is None:
        volumes = [100_000] * n
    return pd.DataFrame({
        "date": dates,
        "open": [round(c * 0.99, 2) for c in closes],
        "high": [round(c * 1.02, 2) for c in closes],
        "low": [round(c * 0.98, 2) for c in closes],
        "close": closes,
        "volume": volumes,
    })


def _set_bar(df: pd.DataFrame, idx: int, **kwargs: Any) -> None:
    """Override one or more fields on an existing weekly bar (in-place)."""
    for k, v in kwargs.items():
        df.at[idx, k] = v


# ===========================================================================
# MomentumReversal13 — pivot branch unit tests (_find_big1)
# ===========================================================================

class TestMr13FindBig1:
    """Direct tests for the _find_big1 pivot-anchor logic.

    These exercise the three core pivot branches without running the full
    scan pipeline.
    """

    @pytest.fixture
    def strategy(self) -> MomentumReversal13Strategy:
        return MomentumReversal13Strategy()

    @pytest.fixture
    def weekly(self) -> pd.DataFrame:
        """30-week declining series used as the base for all pivot tests."""
        return _make_weekly([100.0 - i * 0.5 for i in range(30)])

    def test_pivot_bullish_becomes_big1(self, strategy: MomentumReversal13Strategy,
                                        weekly: pd.DataFrame) -> None:
        """
        Branch 1: the pivot bar (lowest low) is bullish → pivot IS big1.

        Window [10, 20), lowest low at idx 15.  Make idx 15 bullish.
        """
        _set_bar(weekly, 15, low=50.0, open=51.0, close=55.0)
        result = strategy._find_big1(weekly, 10, 20)
        assert result is not None
        big1_idx, pivot_idx = result
        assert pivot_idx == 15
        assert big1_idx == 15, "pivot is bullish so it should BE big1"

    def test_bearish_pivot_new_close_low_uses_right_bullish(
            self, strategy: MomentumReversal13Strategy,
            weekly: pd.DataFrame) -> None:
        """
        Branch 2: pivot is bearish & its close is the window's lowest close
        → big1 = first bullish bar to the RIGHT of the pivot.
        """
        _set_bar(weekly, 15, low=50.0, open=58.0, close=52.0)  # bearish
        _set_bar(weekly, 16, open=52.0, close=56.0)             # right bullish
        result = strategy._find_big1(weekly, 10, 20)
        assert result is not None
        big1_idx, pivot_idx = result
        assert pivot_idx == 15
        assert big1_idx == 16, "pivot is bearish with new-low close → right bullish"

    def test_bearish_pivot_no_new_close_low_uses_left_bullish(
            self, strategy: MomentumReversal13Strategy,
            weekly: pd.DataFrame) -> None:
        """
        Branch 3: pivot is bearish but its close is NOT the lowest close
        → big1 = nearest bullish bar to the LEFT of the pivot.
        """
        _set_bar(weekly, 15, low=50.0, open=58.0, close=56.0)  # bearish
        _set_bar(weekly, 14, close=48.0)                         # even lower close
        _set_bar(weekly, 13, open=53.0, close=57.0)              # left bullish
        result = strategy._find_big1(weekly, 10, 20)
        assert result is not None
        big1_idx, pivot_idx = result
        assert pivot_idx == 15
        assert big1_idx == 13, "pivot is bearish without new-low close → left bullish"


# ===========================================================================
# MomentumReversal13 — full scan tests
# ===========================================================================

def _mr13_base_weekly() -> pd.DataFrame:
    """Build a 30-week DataFrame that triggers a full mr13 match.

    Structure (0-indexed):
        idx  0-16 : pre-downtrend (stable close=60)
        idx 17-26 : 10-week downtrend  (60 → 51)
        idx   26   : big1   — bullish, lowest low, volume above MA20
        idx   27   : small1 — close > big1.close
        idx   28   : small2 — close > big1.close
        idx   29   : small3 — bullish, close breaks out above range
    """
    n = 30
    closes = [60.0] * 17
    closes.extend([60.0 - i for i in range(10)])   # idx 17-26: 60 → 51
    closes.extend([55.0, 58.0, 65.0])               # idx 27-29: small1-3

    weekly = _make_weekly(closes)

    # big1 (idx 26): lowest low, bullish
    _set_bar(weekly, 26, open=48.0, high=52.0, low=46.0, close=51.0,
             volume=1_000_000)

    # small1 (idx 27)
    _set_bar(weekly, 27, open=53.0, high=56.0, low=52.0, close=55.0)

    # small2 (idx 28)
    _set_bar(weekly, 28, open=56.0, high=60.0, low=55.0, close=58.0)

    # small3 (idx 29): bullish + breakout
    _set_bar(weekly, 29, open=60.0, high=66.0, low=59.0, close=65.0)

    return weekly


def _run_mr13_scan(weekly: pd.DataFrame) -> Any:
    """Convenience: instantiate strategy & scan with precomputed weekly."""
    from strategy_runtime import StrategyResult
    s = MomentumReversal13Strategy()
    return s.scan("TEST", pd.DataFrame(), _ctx(), precomputed_weekly=weekly)


class TestMr13Scan:
    """Full scan tests for MomentumReversal13Strategy.

    All tests derive from the same base weekly data (``_mr13_base_weekly``)
    and tweak a single condition so the failure reason is unambiguous.
    """

    def test_full_match(self) -> None:
        """All conditions satisfied → matched & reason_code='matched'."""
        weekly = _mr13_base_weekly()
        result = _run_mr13_scan(weekly)

        assert result.matched, f"expected match, got {result.reason_code}"
        assert result.reason_code == "matched"
        d = result.details
        assert d["signal_type"] == "momentum_reversal_13"
        assert d["big1"]["date"] == str(weekly.at[26, "date"])
        assert d["small1"]["date"] == str(weekly.at[27, "date"])
        assert d["small2"]["date"] == str(weekly.at[28, "date"])
        assert d["small3"]["date"] == str(weekly.at[29, "date"])
        assert d["signal_date"] == str(weekly.at[29, "date"])
        assert d["validation"]["breakout_confirmed"] is True
        assert d["validation"]["all_above_big1"] is True
        assert d["validation"]["big1_volume_above_ma20"] is True

    def test_closes_not_above_big1(self) -> None:
        """
        One of the small-bar closes is ≤ big1.close → no match.

        Setting small3.close = 50 while big1.close = 51 triggers
        reason_code 'closes_not_above_big1'.
        """
        weekly = _mr13_base_weekly()
        _set_bar(weekly, 29, close=50.0, open=49.0)  # still bullish, but ≤ big1.close
        result = _run_mr13_scan(weekly)

        assert not result.matched
        assert result.reason_code == "closes_not_above_big1"

    def test_small3_not_bullish(self) -> None:
        """
        small3 is bearish → no match.

        Setting small3.open > small3.close triggers reason_code 'small3_not_bullish'.
        """
        weekly = _mr13_base_weekly()
        _set_bar(weekly, 29, open=66.0)  # 66 > close=65 → bearish
        result = _run_mr13_scan(weekly)

        assert not result.matched
        assert result.reason_code == "small3_not_bullish"

    def test_small3_no_breakout(self) -> None:
        """
        small3.close does not exceed the high range of [big1 … small2] → no match.

        Setting small3.close = 59 (max high range = 60) triggers
        reason_code 'no_breakout'.
        """
        weekly = _mr13_base_weekly()
        _set_bar(weekly, 29, close=59.0, open=57.0)  # bullish but no breakout
        result = _run_mr13_scan(weekly)

        assert not result.matched
        assert result.reason_code == "no_breakout"

    def test_big1_volume_below_ma20(self) -> None:
        """
        big1 volume is ≤ its 20-week MA → no match.

        Setting big1.volume = 50_000 (MA20 ≈ 100_000) triggers
        reason_code 'big1_volume_not_above_ma20'.
        """
        weekly = _mr13_base_weekly()
        _set_bar(weekly, 26, volume=50_000)
        result = _run_mr13_scan(weekly)

        assert not result.matched
        assert result.reason_code == "big1_volume_not_above_ma20"

    def test_insufficient_weekly_bars(self) -> None:
        """Fewer than min_weekly_bars → reason_code 'insufficient_weekly_bars'."""
        weekly = _make_weekly([60.0] * 20)  # only 20 weeks (min = 24)
        result = _run_mr13_scan(weekly)
        assert not result.matched
        assert result.reason_code == "insufficient_weekly_bars"


# ===========================================================================
# BlackHorse — full scan tests
# ===========================================================================

def _run_bh_scan(weekly: pd.DataFrame) -> Any:
    s = BlackHorseStrategy()
    return s.scan("TEST", pd.DataFrame(), _ctx(), precomputed_weekly=weekly)


class TestBlackHorse:
    """Full scan tests for BlackHorseStrategy.

    The strategy checks the latest 3 completed weekly bars:
      - all bullish   (close > open)
      - body gain strictly increasing
      - volume  strictly increasing
    """

    def test_full_match(self) -> None:
        """3 bullish bars with increasing gains & increasing volumes → matched."""
        closes = [50.0] * 9 + [52.0, 55.0, 60.0]
        volumes = [100_000] * 9 + [100_000, 120_000, 150_000]
        weekly = _make_weekly(closes, volumes)
        # Set explicit opens for strictly increasing body gains:
        #   gain1 = (52-51)/51 ≈ 0.0196
        #   gain2 = (55-52)/52 ≈ 0.0577
        #   gain3 = (60-55)/55 ≈ 0.0909
        _set_bar(weekly, 9,  open=51.0, close=52.0)
        _set_bar(weekly, 10, open=52.0, close=55.0)
        _set_bar(weekly, 11, open=55.0, close=60.0)
        result = _run_bh_scan(weekly)

        assert result.matched, f"expected match, got {result.reason_code}"
        assert result.reason_code == "matched"
        d = result.details
        assert d["signal_type"] == "black_horse_ready"
        assert d["week_1_body_gain"] < d["week_2_body_gain"] < d["week_3_body_gain"]
        assert d["week_1_volume"] < d["week_2_volume"] < d["week_3_volume"]

    def test_not_bullish(self) -> None:
        """One bearish bar → reason_code 'weekly_candle_not_bullish'."""
        closes = [50.0] * 9 + [48.0, 55.0, 60.0]
        volumes = [100_000] * 12
        weekly = _make_weekly(closes, volumes)
        # First bar bearish: open > close
        _set_bar(weekly, 9, open=50.0, close=48.0)   # gain=(48-50)/50=-0.04 < 0
        # Remaining two bars bullish with increasing gains (required to reach the bullish check)
        _set_bar(weekly, 10, open=52.0, close=55.0)  # gain≈0.0577
        _set_bar(weekly, 11, open=55.0, close=60.0)  # gain≈0.0909
        result = _run_bh_scan(weekly)

        assert not result.matched
        assert result.reason_code == "weekly_candle_not_bullish"

    def test_body_gain_not_expanding(self) -> None:
        """
        Body gains are not strictly increasing
        → reason_code 'body_gain_not_expanding'.

        Week 2 gain (0.0385) is LESS than week 1 gain (0.04).
        """
        closes = [50.0] * 9 + [52.0, 54.0, 58.0]
        volumes = [100_000] * 9 + [100_000, 120_000, 150_000]
        weekly = _make_weekly(closes, volumes)
        # Override opens to produce non-increasing gains:
        #   week 1 (idx 9):  open=50, close=52  → gain=0.04
        #   week 2 (idx 10): open=52, close=54  → gain≈0.0385  ← lower!
        #   week 3 (idx 11): open=54, close=58  → gain≈0.0741
        _set_bar(weekly, 9,  open=50.0, close=52.0)
        _set_bar(weekly, 10, open=52.0, close=54.0)
        _set_bar(weekly, 11, open=54.0, close=58.0)
        result = _run_bh_scan(weekly)

        assert not result.matched
        assert result.reason_code == "body_gain_not_expanding"

    def test_volume_not_expanding(self) -> None:
        """
        Volumes are not strictly increasing
        → reason_code 'weekly_volume_not_expanding'.

        Week 3 volume (110_000) is LESS than week 2 volume (120_000).
        """
        closes = [50.0] * 9 + [52.0, 55.0, 60.0]
        volumes = [100_000] * 9 + [100_000, 120_000, 110_000]  # last volume dips
        weekly = _make_weekly(closes, volumes)
        # Gains must be strictly increasing so we reach the volumes check
        _set_bar(weekly, 9,  open=51.0, close=52.0)   # gain≈0.0196
        _set_bar(weekly, 10, open=52.0, close=55.0)   # gain≈0.0577
        _set_bar(weekly, 11, open=55.0, close=60.0)   # gain≈0.0909
        result = _run_bh_scan(weekly)

        assert not result.matched
        assert result.reason_code == "weekly_volume_not_expanding"

    def test_insufficient_weekly_bars(self) -> None:
        """Fewer than min_weekly_bars → reason_code 'insufficient_weekly_bars'."""
        weekly = _make_weekly([50.0] * 8)  # only 8 weeks (min = 12)
        result = _run_bh_scan(weekly)
        assert not result.matched
        assert result.reason_code == "insufficient_weekly_bars"
