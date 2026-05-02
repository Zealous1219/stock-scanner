"""动量反转13周策略 (MomentumReversal13)。

检测流程：
  1. 向前搜索 10 周下跌窗口（连续 10 周内多数周下跌 + 整体收盘价走低）
  2. 在窗口内通过 pivot anchor 模型定位 big1（关键阳线）
     - pivot_bar = 窗口内最低 low 的周
     - pivot_bar 阳线 → big1 = pivot_bar 自身
     - pivot_bar 阴线 + 收盘创新低 → big1 = 右侧第一根阳线
     - pivot_bar 阴线 + 收盘未创新低 → big1 = 左侧最近阳线
  3. big1 后连续三根周线 (small1/2/3) 必须满足：
     - 三收盘 > big1.close
     - small3 阳线
     - small3.close > max(big1.high, small1.high, small2.high)
     - big1.volume > big1 前 20 周成交量 MA20

作者: zealous
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from data_utils import get_completed_weekly_bars
from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult
from strategies.base import BaseStrategy


class MomentumReversal13Strategy(BaseStrategy):
    """检测经历至少 10 周下跌后的动量反转信号。"""

    DEFAULT_PARAMS = {
        "min_downtrend_weeks": 10,
        "min_weekly_bars": 24,
        "reversal_weeks": 3,
    }

    def _init_strategy(self) -> None:
        self.min_downtrend_weeks = int(self.params.get("min_downtrend_weeks", 10))
        self.min_weekly_bars = int(self.params.get("min_weekly_bars", 20))
        self.reversal_weeks = int(self.params.get("reversal_weeks", 3))

    def _validate_params(self) -> None:
        if self.min_downtrend_weeks < 5:
            raise ValueError("min_downtrend_weeks 必须 >= 5")
        if self.reversal_weeks != 3:
            raise ValueError("目前只支持 3 周反转确认")
        min_required = 20 + 1 + self.reversal_weeks
        if self.min_weekly_bars < min_required:
            raise ValueError(f"min_weekly_bars ({self.min_weekly_bars}) 必须 >= {min_required}")

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(
            should_run=True,
            reason_code="completed_weekly_bars_only",
            reason_text="使用已完成周线，忽略当前未完成周。",
        )

    def _is_bullish(self, row: pd.Series) -> bool:
        return float(row["close"]) > float(row["open"])

    def _calc_weekly_volume_ma20(self, weekly: pd.DataFrame, idx: int) -> float | None:
        """计算 big1 前 20 周成交量均值（不含 big1 自身）。"""
        if idx < 20:
            return None
        window = weekly.iloc[idx - 20 : idx]
        if not self._is_weekly_window_contiguous(window):
            return None
        return float(window["volume"].mean())

    @staticmethod
    def _is_weekly_window_contiguous(window: pd.DataFrame) -> bool:
        """校验 20 行窗口日历跨度 ≤ 140 天（约 20 周），防 gap。"""
        if len(window) < 2:
            return True
        first = pd.to_datetime(window.iloc[0]["date"])
        last = pd.to_datetime(window.iloc[-1]["date"])
        return (last - first).days <= 19 * 7 + 7  # 140 days

    def _detect_downtrend(self, weekly: pd.DataFrame, end_idx: int) -> bool:
        """检测 [end_idx - min_downtrend_weeks, end_idx) 窗口内是否存在下跌趋势。

        判定条件：末收盘 < 首收盘（整体下降）且下跌周占比 ≥ 50%。
        """
        if end_idx < self.min_downtrend_weeks:
            return False
        start_idx = end_idx - self.min_downtrend_weeks
        window = weekly.iloc[start_idx:end_idx]
        closes = window["close"].tolist()
        overall_decline = closes[-1] < closes[0]
        down_weeks = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
        down_ratio = down_weeks / max(len(closes) - 1, 1)
        return overall_decline and down_ratio >= 0.5

    def _find_big1(
        self, weekly: pd.DataFrame, start_idx: int, end_idx: int
    ) -> tuple[int, int] | None:
        """在下跌窗口内通过 pivot anchor 模型定位 big1。

        Returns: (big1_idx, pivot_idx) 或 None。
        """
        if start_idx < 0 or end_idx > len(weekly) or start_idx >= end_idx:
            return None
        downtrend_window = weekly.iloc[start_idx:end_idx]
        min_low = downtrend_window["low"].min()
        pivot_idx = None
        for i in range(start_idx, end_idx):
            if weekly.iloc[i]["low"] == min_low:
                pivot_idx = i
                break
        if pivot_idx is None:
            return None
        pivot_bar = weekly.iloc[pivot_idx]
        is_pivot_bullish = self._is_bullish(pivot_bar)

        if is_pivot_bullish:
            return (pivot_idx, pivot_idx)
        else:
            min_close = downtrend_window["close"].min()
            if pivot_bar["close"] == min_close:
                for i in range(pivot_idx + 1, end_idx):
                    if self._is_bullish(weekly.iloc[i]):
                        return (i, pivot_idx)
            else:
                for i in range(pivot_idx - 1, start_idx - 1, -1):
                    if self._is_bullish(weekly.iloc[i]):
                        return (i, pivot_idx)
        return None

    def _build_result_details(
        self,
        weekly: pd.DataFrame,
        big1_idx: int,
        small1_idx: int,
        small2_idx: int,
        small3_idx: int,
        downtrend_weeks: int,
        pivot_idx: int | None = None,
    ) -> Dict[str, Any]:
        big1 = weekly.iloc[big1_idx]
        small1 = weekly.iloc[small1_idx]
        small2 = weekly.iloc[small2_idx]
        small3 = weekly.iloc[small3_idx]
        high_range = weekly.iloc[big1_idx : small2_idx + 1]["high"].max()
        big1_volume_ma20 = self._calc_weekly_volume_ma20(weekly, big1_idx)

        def calc_change(row: pd.Series) -> float:
            o, c = float(row["open"]), float(row["close"])
            return 0.0 if o <= 0 else (c - o) / o

        pivot_info = None
        if pivot_idx is not None:
            pb = weekly.iloc[pivot_idx]
            pivot_info = {
                "date": str(pb["date"]),
                "open": float(pb["open"]),
                "high": float(pb["high"]),
                "low": float(pb["low"]),
                "close": float(pb["close"]),
                "volume": float(pb["volume"]),
                "is_bullish": self._is_bullish(pb),
            }

        result = {
            "signal_type": "momentum_reversal_13",
            "signal_date": str(small3["date"]),
            "downtrend_weeks": downtrend_weeks,
            "big1": {
                "date": str(big1["date"]),
                "open": float(big1["open"]),
                "high": float(big1["high"]),
                "low": float(big1["low"]),
                "close": float(big1["close"]),
                "volume": float(big1["volume"]),
                "volume_ma20": big1_volume_ma20,
                "is_bullish": self._is_bullish(big1),
            },
            "small1": {
                "date": str(small1["date"]),
                "open": float(small1["open"]),
                "high": float(small1["high"]),
                "low": float(small1["low"]),
                "close": float(small1["close"]),
                "volume": float(small1["volume"]),
                "change_pct": round(calc_change(small1), 4),
            },
            "small2": {
                "date": str(small2["date"]),
                "open": float(small2["open"]),
                "high": float(small2["high"]),
                "low": float(small2["low"]),
                "close": float(small2["close"]),
                "volume": float(small2["volume"]),
                "change_pct": round(calc_change(small2), 4),
            },
            "small3": {
                "date": str(small3["date"]),
                "open": float(small3["open"]),
                "high": float(small3["high"]),
                "low": float(small3["low"]),
                "close": float(small3["close"]),
                "volume": float(small3["volume"]),
                "change_pct": round(calc_change(small3), 4),
                "is_bullish": self._is_bullish(small3),
            },
            "validation": {
                "big1_to_small2_high": float(high_range),
                "small3_close": float(small3["close"]),
                "breakout_confirmed": float(small3["close"]) > float(high_range),
                "all_above_big1": all(
                    float(row["close"]) > float(big1["close"])
                    for row in [small1, small2, small3]
                ),
                "big1_volume_above_ma20": (
                    big1_volume_ma20 is not None
                    and float(big1["volume"]) > big1_volume_ma20
                ),
                "big1_is_bullish": self._is_bullish(big1),
            },
        }
        if pivot_info:
            result["pivot_bar"] = pivot_info
        return result

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: StrategyContext,
        precomputed_weekly: pd.DataFrame | None = None,
    ) -> StrategyResult:
        if precomputed_weekly is not None:
            weekly = precomputed_weekly
        else:
            weekly = get_completed_weekly_bars(df, now=context.now)

        if len(weekly) < self.min_weekly_bars:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_weekly_bars",
                reason_text=f"需至少 {self.min_weekly_bars} 周，当前 {len(weekly)} 周",
                details={"available_weeks": int(len(weekly))},
            )

        total_weeks = len(weekly)
        small3_idx = total_weeks - 1
        small2_idx = small3_idx - 1
        small1_idx = small2_idx - 1
        big1_candidate_idx = small1_idx - 1

        if big1_candidate_idx < 0:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_structure_data",
                reason_text="数据不足以形成反转结构。",
                details={"available_weeks": int(len(weekly))},
            )

        found_signal = False
        best_result = None
        search_start = self.min_downtrend_weeks
        search_end = total_weeks - 3

        for downtrend_end in range(search_start, search_end + 1):
            if not self._detect_downtrend(weekly, downtrend_end):
                continue

            window_start = downtrend_end - self.min_downtrend_weeks
            window_end = downtrend_end

            big1_result = self._find_big1(weekly, window_start, window_end)
            if big1_result is None:
                continue

            big1_idx, pivot_idx = big1_result
            expected_small1 = big1_idx + 1
            expected_small2 = big1_idx + 2
            expected_small3 = big1_idx + 3

            if (
                expected_small1 != small1_idx
                or expected_small2 != small2_idx
                or expected_small3 != small3_idx
            ):
                continue

            downtrend_weeks = self.min_downtrend_weeks
            big1 = weekly.iloc[big1_idx]
            small1 = weekly.iloc[small1_idx]
            small2 = weekly.iloc[small2_idx]
            small3 = weekly.iloc[small3_idx]

            all_above_big1 = all(
                float(row["close"]) > float(big1["close"])
                for row in [small1, small2, small3]
            )

            if not all_above_big1:
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "closes_not_above_big1"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="closes_not_above_big1",
                    reason_text="small1/2/3 收盘未全部高于 big1。",
                    details=details,
                )
                continue

            if not self._is_bullish(small3):
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "small3_not_bullish"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="small3_not_bullish",
                    reason_text="small3 不是阳线。",
                    details=details,
                )
                continue

            high_range = weekly.iloc[big1_idx : small2_idx + 1]["high"].max()
            breakout_confirmed = float(small3["close"]) > float(high_range)

            if not breakout_confirmed:
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "no_breakout"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="no_breakout",
                    reason_text=f"small3 收盘 {small3['close']:.2f} 未突破 {high_range:.2f}。",
                    details=details,
                )
                continue

            big1_volume_ma20 = self._calc_weekly_volume_ma20(weekly, big1_idx)
            if big1_volume_ma20 is None:
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "insufficient_volume_ma20_data"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="insufficient_volume_ma20_data",
                    reason_text="big1 前不足 20 周，无法计算成交量 MA20。",
                    details=details,
                )
                continue

            volume_above_ma20 = float(big1["volume"]) > big1_volume_ma20
            if not volume_above_ma20:
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "big1_volume_not_above_ma20"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="big1_volume_not_above_ma20",
                    reason_text=f"big1 成交量未高于 MA20。",
                    details=details,
                )
                continue

            details = self._build_result_details(
                weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
            )
            details["symbol"] = symbol
            return StrategyResult(
                matched=True,
                reason_code="matched",
                reason_text=f"动量反转信号确认。small3 ({small3['date']}) 收盘确认突破。",
                details=details,
            )

        if best_result:
            return best_result
        return StrategyResult(
            matched=False,
            reason_code="no_downtrend_detected",
            reason_text="未检测到满足条件的下跌趋势。",
            details={"available_weeks": int(len(weekly))},
        )

    def compute(self, df: pd.DataFrame) -> bool:
        context = StrategyContext(now=pd.Timestamp.now().to_pydatetime(), stock_pool="unknown")
        return self.scan("unknown", df, context).matched

    @property
    def supported_timeframes(self) -> List[str]:
        return ["weekly", "daily"]


Strategy = MomentumReversal13Strategy
