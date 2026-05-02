"""
动量反转13周策略 (MomentumReversal13)

策略逻辑:
检测经历长期下跌后的动量反转信号。

=== 策略规则 ===

1. 先检测至少10周的下跌趋势
   - 下跌趋势定义: 连续10周或更多周内，收盘价持续走低或整体呈下降态势

2. big1 通过 pivot anchor 模型定位，是下跌趋势中的一根关键阳线
   - big1 必须是通过 pivot anchor 模型（最低 low → 阴阳线分支）定位的关键阳线
   - big1 的成交量必须高于周线成交量 MA20
   - MA20 使用 big1 之前 20 周的成交量均值计算（不含 big1 自身）

3. small1, small2, small3 是 big1 之后连续3个已完成的周线
   - 三者必须按时间顺序排列

4. 三个收盘价都必须高于 big1 的收盘价
   - small1.close > big1.close
   - small2.close > big1.close
   - small3.close > big1.close

5. small3 必须是阳线（收盘价高于开盘价）

6. small3 的收盘价必须突破 big1 到 small2 期间的最高价
   - small3.close > max(big1.high, small1.high, small2.high)

7. 信号只在 small3 收盘后确认
   - 使用已完成周线，不含当前未完成周

8. 信号日期为 small3.date

=== 模糊规则说明 ===

- "下跌趋势"的定义存在模糊性：
  本策略采用简化定义：在至少10周窗口内，收盘价的局部趋势呈下降态势。
  具体实现：检查最近N周窗口，寻找收盘价连续下跌或整体下降的模式。
  安全默认：如果没有明确的下跌趋势证据，策略不触发信号。

- "big1" 通过 pivot anchor 模型定位：
  1. 首先在下跌趋势窗口内找到最低 low 的周，作为 pivot_bar
  2. 如果 pivot_bar 是阳线（收盘 > 开盘）：
     → big1 = pivot_bar 自身
  3. 如果 pivot_bar 是阴线：
     a. 如果 pivot_bar 的收盘价创了窗口内收盘价新低：
        → big1 = pivot_bar 右侧第一根阳线
     b. 如果 pivot_bar 的收盘价没有创新低：
        → big1 = pivot_bar 左侧最近的一根阳线
  4. 如果按上述规则找不到符合条件的阳线，该窗口不触发信号

  这个模型的核心思想是：big1 不是单纯的最低点，而是在下跌动量
  衰竭区域定位的"关键阳线"——可能就在最低点（若最低点是阳线），
  也可能紧接在最低点之后（若最低点是收盘价创新低的阴线），
  也可能在最低点之前（若最低点是收盘价未创新低的阴线）。

- "周线成交量 MA20" 的计算方式：
  使用已完成周线数据，计算 big1 之前 20 周的成交量均值。
  MA20 不包含 big1 自身的成交量，而是用 [idx-20, idx-1] 这 20 周的数据。
  如果 big1 位置之前不足20周数据，视为无法判断，策略不触发信号（安全默认）。

日期: 2026-04-23
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from data_utils import get_completed_weekly_bars
from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult
from strategies.base import BaseStrategy


class MomentumReversal13Strategy(BaseStrategy):
    """
    动量反转13周策略

    检测经历至少10周下跌后的动量反转信号。
    信号由 big1（最低点）和后续三个连续周线 small1, small2, small3 构成。
    """

    DEFAULT_PARAMS = {
        "min_downtrend_weeks": 10,  # 最少下跌趋势周数
        "min_weekly_bars": 24,  # 最少需要的周线数量（big1前20周 + big1 + small1/2/3）
        "reversal_weeks": 3,  # 反转确认周数 (small1, small2, small3)
    }

    def _init_strategy(self) -> None:
        """
        策略参数初始化

        初始化下跌趋势检测参数和反转确认参数。
        """
        self.min_downtrend_weeks = int(self.params.get("min_downtrend_weeks", 10))
        self.min_weekly_bars = int(self.params.get("min_weekly_bars", 20))
        self.reversal_weeks = int(self.params.get("reversal_weeks", 3))

    def _validate_params(self) -> None:
        """
        参数有效性校验

        确保参数组合逻辑正确。
        """
        if self.min_downtrend_weeks < 5:
            raise ValueError("min_downtrend_weeks 必须至少为5周")

        if self.reversal_weeks != 3:
            raise ValueError("reversal_weeks 目前只支持3周反转确认")

        # 最小样本要求：big1 前 20 周（计算 volume MA20）+ big1 + small1/2/3 = 24 周
        min_required_weeks = 20 + 1 + self.reversal_weeks

        if self.min_weekly_bars < min_required_weeks:
            raise ValueError(
                f"min_weekly_bars ({self.min_weekly_bars}) 必须 >= {min_required_weeks} 周 "
                f"（big1 前 20 周 MA20 + big1 + {self.reversal_weeks} 周反转确认）"
            )

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        """
        运行时执行判断

        该策略只使用已完成的周线，不依赖未完成的当前周数据。

        参数:
            context: 运行时上下文

        返回值:
            StrategyDecision: 策略执行决策
        """
        return StrategyDecision(
            should_run=True,
            reason_code="completed_weekly_bars_only",
            reason_text=(
                "MomentumReversal13 策略使用已完成的周线数据，"
                "忽略当前未完成的周线。"
            ),
        )

    def _is_bullish(self, row: pd.Series) -> bool:
        """
        判断周线是否为阳线

        阳线定义: 收盘价 > 开盘价

        参数:
            row: 单周数据行

        返回值:
            bool: 是否为阳线
        """
        return float(row["close"]) > float(row["open"])

    def _calc_weekly_volume_ma20(self, weekly: pd.DataFrame, idx: int) -> float | None:
        """
        计算周线成交量20日均值（MA20）

        使用 big1 之前 20 周的成交量计算均值，不包含 big1 自身的成交量。
        如果 big1 位置之前的数据不足20周，返回 None 表示无法判断。
        如果 20 周窗口在日历上不连续（存在明显 gap），也返回 None，
        避免将过于久远的数据混入均值计算。

        参数:
            weekly: 周线数据
            idx: big1 的位置索引

        返回值:
            float | None: MA20(volume) 值，数据不足或不连续时返回 None
        """
        if idx < 20:  # big1 之前需要至少 20 周数据
            return None

        # 计算 big1 之前 20 周的成交量均值（不含 big1 自身）
        # 使用 idx-20 到 idx-1 这 20 周的数据
        window = weekly.iloc[idx - 20 : idx]

        # 周连续性校验：20 行窗口的日历跨度不应超过 140 天
        # （理论跨度 133 天 = 19 周间隔，额外容忍约 1 周）
        if not self._is_weekly_window_contiguous(window):
            return None

        ma20 = window["volume"].mean()

        return float(ma20)

    @staticmethod
    def _is_weekly_window_contiguous(window: pd.DataFrame) -> bool:
        """
        校验一组周线窗口的连续性。

        通过窗口内最早和最晚的周结束日期（date 列）计算日历跨度。
        对 20 行窗口最大允许 140 天（19*7 + 7 容忍），
        超过此值说明中间缺了若干自然周，不应视为连续 20 周。

        参数:
            window: 周线 DataFrame 切片

        返回值:
            bool: 连续性校验通过返回 True
        """
        if len(window) < 2:
            return True

        first_date = pd.to_datetime(window.iloc[0]["date"])
        last_date = pd.to_datetime(window.iloc[-1]["date"])
        calendar_span = (last_date - first_date).days

        # 对 20 行窗口：19 个周间隔 * 7 天 + 7 天容忍 = 140 天
        max_allowed_span = 19 * 7 + 7  # 140 天

        return calendar_span <= max_allowed_span

    def _detect_downtrend(self, weekly: pd.DataFrame, end_idx: int) -> bool:
        """
        检测下跌趋势

        检查是否存在从 end_idx 向前 min_downtrend_weeks 周的下跌趋势。
        下跌趋势窗口定义为：[end_idx - min_downtrend_weeks, end_idx)
        即 start_idx = end_idx - min_downtrend_weeks，窗口大小固定为 min_downtrend_weeks 周。

        简化判定：在检测窗口内，整体收盘价呈下降态势，且多数周线收盘价低于前周。

        参数:
            weekly: 周线数据
            end_idx: 检测窗口结束位置索引（不包含）

        返回值:
            bool: 是否存在下跌趋势
        """
        if end_idx < self.min_downtrend_weeks:
            return False

        # 下跌趋势窗口: [end_idx - min_downtrend_weeks, end_idx)
        start_idx = end_idx - self.min_downtrend_weeks
        window = weekly.iloc[start_idx : end_idx]

        # 计算窗口内收盘价的趋势
        closes = window["close"].tolist()

        # 简化下跌趋势判定：
        # 1. 窗口末端收盘价低于窗口起始收盘价（整体下降）
        # 2. 窗口内下跌周数（本周收盘低于上周收盘）占比超过50%
        overall_decline = closes[-1] < closes[0]

        down_weeks = 0
        for i in range(1, len(closes)):
            if closes[i] < closes[i - 1]:
                down_weeks += 1

        down_ratio = down_weeks / (len(closes) - 1)
        majority_down = down_ratio >= 0.5

        return overall_decline and majority_down

    def _find_big1(self, weekly: pd.DataFrame, start_idx: int, end_idx: int) -> tuple[int, int] | None:
        """
        寻找下跌趋势窗口内的最低点 big1（使用 pivot anchor 模型）

        Pivot anchor 模型:
        1. 首先找到 pivot_bar = 下跌趋势窗口内最低 low 的周
        2. big1 必须始终是阳线（bullish）

        规则:
        - 如果 pivot_bar 是阳线: big1 = pivot_bar
        - 如果 pivot_bar 是阴线:
          - 如果 pivot_bar.close 是下跌趋势窗口内最低的收盘价:
            - big1 = pivot_bar 右侧第一个阳线（必须存在）
          - 否则:
            - big1 = pivot_bar 左侧最近的一个阳线（必须存在）

        如果找不到符合条件的阳线作为 big1，返回 None。

        参数:
            weekly: 周线数据
            start_idx: 下跌趋势窗口起始索引（包含）
            end_idx: 下跌趋势窗口结束索引（不包含）

        返回值:
            tuple[int, int] | None: (big1_idx, pivot_idx) 元组，未找到则返回 None
        """
        if start_idx < 0 or end_idx > len(weekly) or start_idx >= end_idx:
            return None

        # 1. 找到 pivot_bar = 下跌趋势窗口内最低 low 的周
        downtrend_window = weekly.iloc[start_idx:end_idx]
        min_low = downtrend_window["low"].min()

        # 找到第一个最低 low 的位置（时间最早）
        pivot_idx = None
        for i in range(start_idx, end_idx):
            if weekly.iloc[i]["low"] == min_low:
                pivot_idx = i
                break

        if pivot_idx is None:
            return None

        pivot_bar = weekly.iloc[pivot_idx]
        is_pivot_bullish = self._is_bullish(pivot_bar)

        # 2. 根据规则确定 big1
        if is_pivot_bullish:
            # 规则1: pivot_bar 是阳线，big1 = pivot_bar
            return (pivot_idx, pivot_idx)
        else:
            # pivot_bar 是阴线
            # 检查 pivot_bar.close 是否是下跌趋势窗口内最低的收盘价
            min_close_in_window = downtrend_window["close"].min()
            is_pivot_close_lowest = pivot_bar["close"] == min_close_in_window

            if is_pivot_close_lowest:
                # 规则2: pivot_bar.close 是最低收盘价，big1 = pivot_bar 右侧第一个阳线
                # 搜索 pivot_bar 右侧（从 pivot_idx+1 到 end_idx-1）
                for i in range(pivot_idx + 1, end_idx):
                    if self._is_bullish(weekly.iloc[i]):
                        return (i, pivot_idx)
            else:
                # 规则3: pivot_bar.close 不是最低收盘价，big1 = pivot_bar 左侧最近的一个阳线
                # 搜索 pivot_bar 左侧（从 pivot_idx-1 到 start_idx）
                for i in range(pivot_idx - 1, start_idx - 1, -1):
                    if self._is_bullish(weekly.iloc[i]):
                        return (i, pivot_idx)

        # 没有找到符合条件的阳线
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
        """
        构建策略结果详情

        提取关键周线数据用于人工复核。

        参数:
            weekly: 周线数据
            big1_idx: big1 索引
            small1_idx: small1 索引
            small2_idx: small2 索引
            small3_idx: small3 索引
            downtrend_weeks: 下跌趋势周数
            pivot_idx: pivot_bar 索引（可选）

        返回值:
            Dict: 结构化详情字典
        """
        big1 = weekly.iloc[big1_idx]
        small1 = weekly.iloc[small1_idx]
        small2 = weekly.iloc[small2_idx]
        small3 = weekly.iloc[small3_idx]

        # 计算从 big1 到 small2 的最高价
        high_range = weekly.iloc[big1_idx : small2_idx + 1]["high"].max()

        # 计算 big1 的周线成交量 MA20
        big1_volume_ma20 = self._calc_weekly_volume_ma20(weekly, big1_idx)

        # 计算各周的涨跌幅
        def calc_change(row: pd.Series) -> float:
            open_p = float(row["open"])
            close_p = float(row["close"])
            if open_p <= 0:
                return 0.0
            return (close_p - open_p) / open_p

        # 构建 pivot_bar 信息（如果提供）
        pivot_info = None
        if pivot_idx is not None:
            pivot_bar = weekly.iloc[pivot_idx]
            pivot_info = {
                "date": str(pivot_bar["date"]),
                "open": float(pivot_bar["open"]),
                "high": float(pivot_bar["high"]),
                "low": float(pivot_bar["low"]),
                "close": float(pivot_bar["close"]),
                "volume": float(pivot_bar["volume"]),
                "is_bullish": self._is_bullish(pivot_bar),
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
                "all_above_big1": (
                    float(small1["close"]) > float(big1["close"])
                    and float(small2["close"]) > float(big1["close"])
                    and float(small3["close"]) > float(big1["close"])
                ),
                "big1_volume_above_ma20": (
                    big1_volume_ma20 is not None
                    and float(big1["volume"]) > big1_volume_ma20
                ),
                "big1_is_bullish": self._is_bullish(big1),
            },
        }

        # 添加 pivot_bar 信息
        if pivot_info:
            result["pivot_bar"] = pivot_info

        return result

    def scan(
        self, symbol: str, df: pd.DataFrame, context: StrategyContext,
        precomputed_weekly: pd.DataFrame | None = None,
    ) -> StrategyResult:
        """
        单标的完整扫描

        对单个股票执行动量反转13策略扫描，返回结构化结果。

        参数:
            symbol: 股票代码，如 '000001.SZ'
            df: 日线历史数据 DataFrame
            context: 运行时上下文
            precomputed_weekly: replay weekly cache 路径传入的预切片周线数据。
                非 None 时直接使用，跳过 get_completed_weekly_bars。
                普通 scanner 调用不传此参数，保持原有路径不变。

        返回值:
            StrategyResult: 包含是否命中、原因码、描述和详情
        """
        # Weekly-strategy replay optimization v1:
        # Use pre-sliced weekly bars from the replay cache when available.
        if precomputed_weekly is not None:
            weekly = precomputed_weekly
        else:
            weekly = get_completed_weekly_bars(df, now=context.now)

        if len(weekly) < self.min_weekly_bars:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_weekly_bars",
                reason_text=(
                    f"周线数据不足，需要至少 {self.min_weekly_bars} 周，"
                    f"当前仅有 {len(weekly)} 周。"
                ),
                details={"available_weeks": int(len(weekly))},
            )

        # 从最近的周线向前搜索可能的反转信号
        # small3 必须是最新的已完成周线
        total_weeks = len(weekly)
        small3_idx = total_weeks - 1
        small2_idx = small3_idx - 1
        small1_idx = small2_idx - 1
        big1_candidate_idx = small1_idx - 1

        # 检查是否有足够的数据形成反转结构
        if big1_candidate_idx < 0:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_structure_data",
                reason_text="数据不足以形成反转结构（需要至少4周：big1 + small1/2/3）。",
                details={"available_weeks": int(len(weekly))},
            )

        # 向前搜索 big1 和下跌趋势
        # big1 可能是下跌趋势的最后一周或下跌趋势中的最低点
        found_signal = False
        best_result = None

        # 合法的 1+3 结构要求 big1, small1, small2, small3 是最后 4 根周线
        # 即：big1_idx = total_weeks - 4, small1_idx = total_weeks - 3, small2_idx = total_weeks - 2, small3_idx = total_weeks - 1
        #
        # Downtrend window 必须严格在 big1 结束，不能包含 small1/2/3
        # _detect_downtrend() 定义窗口为 [downtrend_end - min_downtrend_weeks, downtrend_end)
        # _find_big1() 在该窗口内搜索最低点 big1
        #
        # 要求：
        # 1. big1 必须在 downtrend window 内: big1_idx >= downtrend_end - min_downtrend_weeks, big1_idx < downtrend_end
        # 2. downtrend window 不包含 small1: downtrend_end <= total_weeks - 3
        # 3. big1_idx 可以等于 total_weeks - 4 (合法结构)，所以 downtrend_end 必须 > total_weeks - 4
        #    因为 _find_big1() 使用半开区间，要求 big1_idx < window_end = downtrend_end
        #
        # 因此 downtrend_end 的范围：
        #   下界：downtrend_end >= min_downtrend_weeks (检测窗口不越界)
        #   上界：downtrend_end <= total_weeks - 3 (不包含 small1)
        #   且 downtrend_end > total_weeks - 4 (让 big1_idx = total_weeks - 4 可达)
        #
        # 综上：downtrend_end ∈ [min_downtrend_weeks, total_weeks - 3]
        # 当 total_weeks - 4 >= min_downtrend_weeks 时，big1_idx = total_weeks - 4 可被包含
        search_start = self.min_downtrend_weeks
        search_end = total_weeks - 3

        # 向前搜索可能的下跌窗口位置
        for downtrend_end in range(search_start, search_end + 1):
            # 检测从 downtrend_end 向前是否存在下跌趋势
            if not self._detect_downtrend(weekly, downtrend_end):
                continue

            # 计算下跌趋势窗口的边界
            # 窗口 = [downtrend_end - min_downtrend_weeks, downtrend_end)
            window_start = downtrend_end - self.min_downtrend_weeks
            window_end = downtrend_end

            # 在下跌趋势窗口内寻找最低点 big1
            big1_result = self._find_big1(weekly, window_start, window_end)
            if big1_result is None:
                continue

            big1_idx, pivot_idx = big1_result

            # 验证 big1, small1, small2, small3 的位置关系
            # small1, small2, small3 必须是 big1 之后的连续三周
            expected_small1 = big1_idx + 1
            expected_small2 = big1_idx + 2
            expected_small3 = big1_idx + 3

            # 检查是否与当前的 small1, small2, small3 位置匹配
            if (
                expected_small1 != small1_idx
                or expected_small2 != small2_idx
                or expected_small3 != small3_idx
            ):
                # 只接受 small3 为最新周线的情况
                continue

            downtrend_weeks = self.min_downtrend_weeks

            # 验证反转条件
            big1 = weekly.iloc[big1_idx]
            small1 = weekly.iloc[small1_idx]
            small2 = weekly.iloc[small2_idx]
            small3 = weekly.iloc[small3_idx]

            # 条件1: small1, small2, small3 的收盘价都必须高于 big1 收盘价
            all_above_big1 = (
                float(small1["close"]) > float(big1["close"])
                and float(small2["close"]) > float(big1["close"])
                and float(small3["close"]) > float(big1["close"])
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
                    reason_text=(
                        "small1, small2, small3 的收盘价并非全部高于 big1 收盘价。"
                    ),
                    details=details,
                )
                continue

            # 条件2: small3 必须是阳线
            if not self._is_bullish(small3):
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "small3_not_bullish"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="small3_not_bullish",
                    reason_text="small3 不是阳线（收盘价未高于开盘价）。",
                    details=details,
                )
                continue

            # 条件3: small3 收盘价必须突破 big1 到 small2 的最高价
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
                    reason_text=(
                        f"small3 收盘价 ({small3['close']:.2f}) "
                        f"未突破 big1 到 small2 的最高价 ({high_range:.2f})。"
                    ),
                    details=details,
                )
                continue

            # 条件4: big1 成交量必须高于周线成交量 MA20
            # 使用周线数据计算 MA20，确保数据一致性
            big1_volume_ma20 = self._calc_weekly_volume_ma20(weekly, big1_idx)
            if big1_volume_ma20 is None:
                # 数据不足20周，无法判断，视为不满足条件
                details = self._build_result_details(
                    weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
                )
                details["symbol"] = symbol
                details["failure_reason"] = "insufficient_volume_ma20_data"
                best_result = StrategyResult(
                    matched=False,
                    reason_code="insufficient_volume_ma20_data",
                    reason_text=(
                        f"big1 位置之前的数据不足20周，无法计算成交量 MA20。"
                    ),
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
                    reason_text=(
                        f"big1 成交量 ({float(big1['volume']):,.0f}) "
                        f"未高于周线 MA20 ({big1_volume_ma20:,.0f})。"
                    ),
                    details=details,
                )
                continue

            # 所有条件满足，构建成功结果
            details = self._build_result_details(
                weekly, big1_idx, small1_idx, small2_idx, small3_idx, downtrend_weeks, pivot_idx
            )
            details["symbol"] = symbol

            return StrategyResult(
                matched=True,
                reason_code="matched",
                reason_text=(
                    f"动量反转信号确认：经历 {downtrend_weeks} 周下跌后，"
                    f"small3 ({small3['date']}) 收盘确认反转突破。"
                ),
                details=details,
            )

        # 返回最近失败的原因（如果有）
        if best_result:
            return best_result

        return StrategyResult(
            matched=False,
            reason_code="no_downtrend_detected",
            reason_text="未能检测到满足条件的下跌趋势。",
            details={"available_weeks": int(len(weekly))},
        )

    def compute(self, df: pd.DataFrame) -> bool:
        """
        布尔策略信号（历史兼容接口）

        参数:
            df: 日线历史数据 DataFrame

        返回值:
            bool: 是否命中策略
        """
        context = StrategyContext(
            now=pd.Timestamp.now().to_pydatetime(), stock_pool="unknown"
        )
        return self.scan("unknown", df, context).matched

    @property
    def supported_timeframes(self) -> List[str]:
        """
        支持的时间周期

        该策略专用于周线数据，日线数据需转换为周线后使用。

        返回值:
            List[str]: 支持的时间周期列表
        """
        return ["weekly", "daily"]


# 模块导出约定
Strategy = MomentumReversal13Strategy