"""策略模块初始化，提供策略注册、加载和工厂函数。

内置策略：
  - moving_average / ma      移动均线策略
  - black_horse / bh         黑马周线策略
  - momentum_reversal_13 / mr13  动量反转13周策略

作者: zealous
"""

from strategies.base import BaseStrategy
from strategies.black_horse import BlackHorseStrategy
from strategies.momentum_reversal_13 import MomentumReversal13Strategy
from strategies.moving_average import MovingAverageStrategy
import os
import importlib
import logging

logger = logging.getLogger(__name__)

BUILTIN_STRATEGIES = {
    "moving_average": MovingAverageStrategy,
    "ma": MovingAverageStrategy,
    "black_horse": BlackHorseStrategy,
    "bh": BlackHorseStrategy,
    "momentum_reversal_13": MomentumReversal13Strategy,
    "mr13": MomentumReversal13Strategy,
}


def list_strategies() -> dict:
    """返回所有可用策略名到类的映射。"""
    strategies = BUILTIN_STRATEGIES.copy()
    strategies_dir = os.path.dirname(__file__)
    for filename in os.listdir(strategies_dir):
        if filename.endswith(".py") and not filename.startswith("_") and filename != "base.py":
            module_name = filename[:-3]
            if module_name not in BUILTIN_STRATEGIES:
                strategies[module_name] = module_name
    return strategies


def load_strategy(strategy_name: str, params: dict = None) -> BaseStrategy:
    """动态加载策略，支持内置策略和 strategies/ 下自定义模块。"""
    strategy_name = strategy_name.lower()
    if strategy_name in BUILTIN_STRATEGIES:
        strategy_class = BUILTIN_STRATEGIES[strategy_name]
        logger.info("加载内置策略: %s", strategy_name)
        return strategy_class(params)
    try:
        module = importlib.import_module(f"strategies.{strategy_name}")
        strategy_class = getattr(module, "Strategy")
        logger.info("加载外部策略: %s", strategy_name)
        return strategy_class(params)
    except (ImportError, AttributeError) as e:
        logger.error("加载策略失败: %s, 错误: %s", strategy_name, e)
        available = list(BUILTIN_STRATEGIES.keys())
        raise ValueError(f"策略 '{strategy_name}' 不存在。可用策略: {available}")


def create_strategy_from_config(config: dict) -> BaseStrategy:
    """从配置字典创建策略实例。"""
    name = config.get("name")
    params = config.get("params", {})
    if not name:
        raise ValueError("配置必须包含 'name' 字段")
    return load_strategy(name, params)


__all__ = [
    "BaseStrategy",
    "BlackHorseStrategy",
    "MomentumReversal13Strategy",
    "MovingAverageStrategy",
    "load_strategy",
    "create_strategy_from_config",
    "list_strategies",
]
