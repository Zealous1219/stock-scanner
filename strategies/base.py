"""策略基类。所有策略须继承此类。

初始化链：
  DEFAULT_PARAMS → 用户传入 params → _init_strategy() → _validate_params()

接口调用链：
  scanner / replay  →  can_run(context)     → StrategyDecision  （任务级判断，默认始终允许）
                    →  scan(symbol, df, context, precomputed_weekly=None) → StrategyResult

scan() 的 df 参数要求列：date, open, high, low, close, volume（日线标准格式）。
precomputed_weekly 仅 replay 框架传入（周线策略加速），scanner 始终传 None。

compute(df) 是历史兼容接口：基类 scan() 默认实现调用它，但不会反向。
新策略直接实现 scan() 即可，无需实现 compute()。
"""

from abc import ABC
from typing import Dict, Any, List

from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult


class BaseStrategy(ABC):
    """策略基类。

    初始化链：DEFAULT_PARAMS → 用户 params → _init_strategy() → _validate_params()

    须实现的方法：
      - scan(symbol, df, context)  → StrategyResult  核心扫描逻辑
      - _init_strategy()           推荐实现，从 self.params 提取属性
      - _validate_params()         推荐实现，校验参数合法性

    可选覆盖：
      - can_run(context)      任务级判断，默认始终允许
      - compute(df)           历史兼容接口（新策略不需要）
      - supported_timeframes  声明支持的时间周期
      - precomputed_weekly    周线策略可加的 scan() 额外参数，用于 replay 加速"""

    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any] = None):
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        self._init_strategy()
        self._validate_params()

    def _init_strategy(self):
        """子类覆盖：从 self.params 提取并初始化策略属性。"""
        pass

    def _validate_params(self):
        """子类覆盖：参数合法性校验，无效时抛 ValueError。"""
        pass

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        """任务级执行判断。默认始终允许执行。"""
        return StrategyDecision(
            should_run=True,
            reason_code="always_run",
            reason_text="策略默认可执行",
        )

    def scan(
        self, symbol: str, df, context: StrategyContext
    ) -> StrategyResult:
        """单标的完整扫描。

        默认实现调用 compute() 将布尔结果包装为 StrategyResult。
        子类建议直接覆盖此方法以返回更丰富的结构化结果。
        """
        matched = self.compute(df)
        return StrategyResult(
            matched=matched,
            reason_code="matched" if matched else "not_matched",
            reason_text=f"策略 {self.name} 判定结果: {matched}",
        )

    def compute(self, df) -> bool:
        """布尔策略信号，历史兼容接口。"""
        raise NotImplementedError(
            f"策略 {self.name} 需要实现 compute() 或 scan()"
        )

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]):
        self.params.update(params)
        self._validate_params()
        self._init_strategy()

    @property
    def supported_timeframes(self) -> List[str]:
        return ["daily", "weekly"]

    def is_timeframe_supported(self, timeframe: str) -> bool:
        return timeframe in self.supported_timeframes

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return f"策略: {self.name}"

    def backtest(self, df, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        raise NotImplementedError("回测接口预留")

    def evaluate(self, returns, benchmark=None) -> Dict[str, float]:
        raise NotImplementedError("评估接口预留")

    def combine(self, other_strategy) -> "BaseStrategy":
        raise NotImplementedError("策略组合接口预留")
