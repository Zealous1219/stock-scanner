"""
策略基类模块

定义所有策略必须继承的基类接口。

=== 使用说明 ===

策略运行时接口（推荐模式）：
    1. can_run(context)  -> StrategyDecision
       决定策略是否应在本次任务中执行，例如检查交易时间、市场状态等。

    2. scan(symbol, df, context) -> StrategyResult
       对单个标的执行完整扫描，返回结构化结果（是否命中、原因码、详情）。

关于 compute()：
    compute(df) 是历史接口，仅供已有策略兼容使用。
    基类 scan() 的默认实现会调用 compute()，将布尔结果包装为 StrategyResult。
    已有策略（如 MovingAverageStrategy）只需实现 compute() 即可通过默认 scan() 运行。
    新策略建议直接实现 scan()，此时 compute() 不再需要。
    注意：compute() 不会反向调用 scan()，兼容路径是单向的。

兼容性说明：
    - 如果子类只实现了 compute()，默认 scan() 会调用它并返回 StrategyResult
    - 如果子类实现了 scan()，则覆盖默认实现，compute() 不再被运行时调用
    - 两条路径互不反向调用，不存在递归风险
    - can_run() 默认始终返回 should_run=True，子类可覆盖以添加运行条件

接口规范:
    - __init__: 初始化方法
    - can_run: 运行时执行判断（推荐实现）
    - scan: 单标的完整扫描（推荐实现）
    - compute: 布尔策略信号（历史接口，保留兼容）
    - get_params: 参数配置方法
    - supported_timeframes: 支持的时间周期

作者: AI Assistant
日期: 2026-03-28
"""

from abc import ABC
from typing import Dict, Any, List

from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult


class BaseStrategy(ABC):
    """
    策略基类

    所有策略必须继承此类。

    推荐实现的接口（运行时主合约）：
    - can_run(context): 判断策略是否应执行，返回 StrategyDecision
    - scan(symbol, df, context): 对单标的扫描，返回 StrategyResult

    历史兼容接口：
    - compute(df): 布尔策略信号（仅 scan() 默认实现调用，不会反向调用 scan()）

    可选重写：
    - _init_strategy: 策略初始化（重写时需调用super）
    - _validate_params: 参数校验
    """

    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化策略

        参数:
            params: 策略参数字典
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self._init_strategy()
        self._validate_params()

    def _init_strategy(self):
        """
        策略特定初始化

        子类可重写此方法进行策略特定的初始化操作。
        重写时需要调用 super()._init_strategy()
        """
        pass

    def _validate_params(self):
        """
        参数校验

        子类可重写此方法进行参数校验。
        如果参数无效，应抛出 ValueError。
        """
        pass

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        """
        运行时执行判断（推荐接口）

        判断策略是否应在当前上下文中执行。
        默认实现始终允许执行；子类可覆盖以添加运行条件，
        例如检查交易时间、市场状态等。

        参数:
            context: 运行时上下文，包含当前时间、股票池、配置等

        返回值:
            StrategyDecision: 包含是否执行、原因码和描述
        """
        return StrategyDecision(
            should_run=True,
            reason_code="always_run",
            reason_text="策略默认可执行",
        )

    def scan(self, symbol: str, df, context: StrategyContext) -> StrategyResult:
        """
        单标的完整扫描（推荐接口）

        对单个标的执行策略逻辑并返回结构化结果。

        默认实现：调用 compute() 并将布尔结果包装为 StrategyResult。
        这是一条单向兼容路径：scan() 可调用 compute()，但 compute() 不会反向调用 scan()。

        子类如需更丰富的返回信息，建议直接覆盖此方法。
        覆盖 scan() 后，compute() 不再被运行时调用。

        参数:
            symbol: 股票代码，如 '000001.SZ'
            df: 股票历史数据 DataFrame
            context: 运行时上下文

        返回值:
            StrategyResult: 包含是否命中、原因码、描述和详情
        """
        matched = self.compute(df)
        return StrategyResult(
            matched=matched,
            reason_code="matched" if matched else "not_matched",
            reason_text=f"策略 {self.name} 判定结果: {matched}",
        )

    def compute(self, df) -> bool:
        """
        布尔策略信号（历史接口）

        返回策略是否命中。此方法仅供已有策略兼容使用。
        基类 scan() 的默认实现会调用 compute() 将布尔结果包装为 StrategyResult。

        已有策略只需实现 compute() 即可通过默认 scan() 与运行时对接。
        新策略如果直接实现了 scan()，则 compute() 不再被运行时调用，可不实现。

        注意：此方法不会反向调用 scan()。

        参数:
            df: 股票历史数据DataFrame

        返回值:
            bool: True表示符合策略条件，False表示不符合
        """
        raise NotImplementedError(
            f"策略 {self.name} 需要实现 compute() 或直接实现 scan()"
        )

    def get_params(self) -> Dict[str, Any]:
        """
        获取策略参数

        返回值:
            策略参数字典的副本
        """
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]):
        """
        设置策略参数

        参数:
            params: 新的参数字典
        """
        self.params.update(params)
        self._validate_params()
        self._init_strategy()

    @property
    def supported_timeframes(self) -> List[str]:
        """
        支持的时间周期

        子类可重写此属性声明支持的时间周期。

        返回值:
            支持的时间周期列表，默认支持日线和周线
        """
        return ['daily', 'weekly']

    def is_timeframe_supported(self, timeframe: str) -> bool:
        """
        检查是否支持指定的时间周期

        参数:
            timeframe: 时间周期

        返回值:
            bool: 是否支持
        """
        return timeframe in self.supported_timeframes

    @property
    def name(self) -> str:
        """
        策略名称

        子类可重写此属性。

        返回值:
            策略名称
        """
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """
        策略描述

        子类可重写此属性。

        返回值:
            策略描述
        """
        return f"策略: {self.name}"

    def backtest(self, df, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        回测策略表现

        预留接口，子类可实现。

        参数:
            df: 历史数据
            start_date: 回测开始日期
            end_date: 回测结束日期

        返回值:
            回测结果字典

        异常:
            NotImplementedError: 预留接口，尚未实现
        """
        raise NotImplementedError("回测接口预留")

    def evaluate(self, returns, benchmark=None) -> Dict[str, float]:
        """
        评估策略绩效

        预留接口，子类可实现。

        参数:
            returns: 策略收益序列
            benchmark: 基准收益序列

        返回值:
            绩效指标字典

        异常:
            NotImplementedError: 预留接口，尚未实现
        """
        raise NotImplementedError("评估接口预留")

    def combine(self, other_strategy) -> 'BaseStrategy':
        """
        组合多个策略

        预留接口，子类可实现。

        参数:
            other_strategy: 另一个策略实例

        返回值:
            组合后的策略

        异常:
            NotImplementedError: 预留接口，尚未实现
        """
        raise NotImplementedError("策略组合接口预留")
