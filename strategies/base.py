"""
策略基类模块

定义所有策略必须继承的基类接口。

接口规范:
    - __init__: 初始化方法
    - compute: 策略计算方法
    - get_params: 参数配置方法
    - supported_timeframes: 支持的时间周期

作者: AI Assistant
日期: 2026-03-28
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseStrategy(ABC):
    """
    策略基类

    所有策略必须继承此类并实现以下方法:
    - compute: 策略计算方法（必须实现）
    - _init_strategy: 策略初始化（可选，重写时需调用super）
    - _validate_params: 参数校验（可选）
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

    @abstractmethod
    def compute(self, df) -> bool:
        """
        计算策略信号

        这是策略的核心方法，必须由子类实现。

        参数:
            df: 股票历史数据DataFrame

        返回值:
            bool: True表示符合策略条件，False表示不符合
        """
        pass

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
