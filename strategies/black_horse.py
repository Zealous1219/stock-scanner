"""Black horse weekly preparation strategy."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from data_utils import get_completed_weekly_bars
from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult
from strategies.base import BaseStrategy


class BlackHorseStrategy(BaseStrategy):
    """Find symbols with three increasingly strong completed weekly candles."""

    DEFAULT_PARAMS = {
        "required_weeks": 3,
        "min_weekly_bars": 12,
    }

    def _init_strategy(self) -> None:
        self.required_weeks = int(self.params.get("required_weeks", 3))
        self.min_weekly_bars = int(self.params.get("min_weekly_bars", 12))

    def _validate_params(self) -> None:
        if self.required_weeks != 3:
            raise ValueError("black_horse currently requires exactly 3 completed weekly bars")

        if self.min_weekly_bars < self.required_weeks:
            raise ValueError("min_weekly_bars must be >= required_weeks")

    def _weekly_body_gain(self, row: pd.Series) -> float:
        open_price = float(row["open"])
        close_price = float(row["close"])
        if open_price <= 0:
            return -1.0
        return (close_price - open_price) / open_price

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(
            should_run=True,
            reason_code="completed_weekly_bars_only",
            reason_text=(
                "Black horse strategy uses the latest three completed weekly bars "
                "and ignores the current unfinished week."
            ),
        )

    def _build_result_details(self, latest_weeks: pd.DataFrame) -> Dict[str, Any]:
        enriched = latest_weeks.copy()
        enriched["body_gain"] = enriched.apply(self._weekly_body_gain, axis=1)
        enriched["week_end"] = pd.to_datetime(enriched["date"]).dt.strftime("%Y-%m-%d")

        return {
            "signal_type": "black_horse_ready",
            "latest_week_end": enriched.iloc[-1]["week_end"],
            "week_1_end": enriched.iloc[0]["week_end"],
            "week_2_end": enriched.iloc[1]["week_end"],
            "week_3_end": enriched.iloc[2]["week_end"],
            "week_1_body_gain": round(float(enriched.iloc[0]["body_gain"]), 6),
            "week_2_body_gain": round(float(enriched.iloc[1]["body_gain"]), 6),
            "week_3_body_gain": round(float(enriched.iloc[2]["body_gain"]), 6),
            "week_1_volume": float(enriched.iloc[0]["volume"]),
            "week_2_volume": float(enriched.iloc[1]["volume"]),
            "week_3_volume": float(enriched.iloc[2]["volume"]),
        }

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: StrategyContext,
        precomputed_weekly: pd.DataFrame | None = None,
    ) -> StrategyResult:
        # Weekly-strategy replay optimization v1:
        # If the caller (replay loop) has already pre-sliced the completed
        # weekly bars for this snapshot, use them directly to skip the
        # daily→weekly resample.  The normal scanner path passes None and
        # falls through to the standard get_completed_weekly_bars call.
        if precomputed_weekly is not None:
            weekly = precomputed_weekly
        else:
            weekly = get_completed_weekly_bars(df, now=context.now)

        if len(weekly) < self.min_weekly_bars:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_weekly_bars",
                reason_text="Not enough completed weekly bars to evaluate the black horse strategy.",
                details={"available_weeks": int(len(weekly))},
            )

        latest_weeks = weekly.tail(self.required_weeks).reset_index(drop=True)
        gains = [self._weekly_body_gain(row) for _, row in latest_weeks.iterrows()]
        volumes = latest_weeks["volume"].tolist()

        all_bullish = all(gain > 0 for gain in gains)
        gains_expanding = gains[0] < gains[1] < gains[2]
        volumes_expanding = volumes[0] < volumes[1] < volumes[2]

        details = self._build_result_details(latest_weeks)
        details["symbol"] = symbol

        if not all_bullish:
            return StrategyResult(
                matched=False,
                reason_code="weekly_candle_not_bullish",
                reason_text="At least one of the latest three completed weekly candles is not bullish.",
                details=details,
            )

        if not gains_expanding:
            return StrategyResult(
                matched=False,
                reason_code="body_gain_not_expanding",
                reason_text="The latest three completed weekly body gains are not strictly increasing.",
                details=details,
            )

        if not volumes_expanding:
            return StrategyResult(
                matched=False,
                reason_code="weekly_volume_not_expanding",
                reason_text="The latest three completed weekly volumes are not strictly increasing.",
                details=details,
            )

        return StrategyResult(
            matched=True,
            reason_code="matched",
            reason_text="Symbol is in black horse preparation state based on completed weekly bars.",
            details=details,
        )

    def compute(self, df: pd.DataFrame) -> bool:
        context = StrategyContext(now=pd.Timestamp.now().to_pydatetime(), stock_pool="unknown")
        return self.scan("unknown", df, context).matched

    @property
    def supported_timeframes(self) -> List[str]:
        return ["daily", "weekly"]


Strategy = BlackHorseStrategy
