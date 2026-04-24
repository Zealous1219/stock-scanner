"""
策略模块包

包含所有可用的选股策略和策略加载器。

内置策略:
    - moving_average: 移动均线策略
    - black_horse: 黑马周线策略
    - momentum_reversal_13: 动量反转13周策略

使用示例:
    from strategies import load_strategy

    # 加载指定策略
    strategy = load_strategy('moving_average')

    # 获取所有可用策略列表
    from strategies import list_strategies
    print(list_strategies())

作者: AI Assistant
日期: 2026-03-28
"""

from strategies.base import BaseStrategy
from strategies.black_horse import BlackHorseStrategy
from strategies.momentum_reversal_13 import MomentumReversal13Strategy
from strategies.moving_average import MovingAverageStrategy
import os
import importlib
import logging

logger = logging.getLogger(__name__)

# 内置策略映射
BUILTIN_STRATEGIES = {
    'moving_average': MovingAverageStrategy,
    'ma': MovingAverageStrategy,
    'black_horse': BlackHorseStrategy,
    'bh': BlackHorseStrategy,
    'momentum_reversal_13': MomentumReversal13Strategy,
    'mr13': MomentumReversal13Strategy,
}


def list_strategies() -> dict:
    """
    获取所有可用策略列表

    返回值:
        dict: 策略名称到策略类的映射
    """
    strategies = BUILTIN_STRATEGIES.copy()

    strategies_dir = os.path.dirname(__file__)
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and not filename.startswith('_') and filename != 'base.py':
            module_name = filename[:-3]
            if module_name not in BUILTIN_STRATEGIES:
                strategies[module_name] = module_name

    return strategies


def load_strategy(strategy_name: str, params: dict = None) -> BaseStrategy:
    """
    动态加载策略

    参数:
        strategy_name: 策略名称
            - 'moving_average' 或 'ma': 移动均线策略
            - 其他: 从strategies目录动态导入
        params: 策略参数字典

    返回值:
        BaseStrategy: 策略实例

    异常:
        ValueError: 策略不存在或参数无效
    """
    strategy_name = strategy_name.lower()

    if strategy_name in BUILTIN_STRATEGIES:
        strategy_class = BUILTIN_STRATEGIES[strategy_name]
        logger.info(f"加载内置策略: {strategy_name}")
        return strategy_class(params)

    try:
        module = importlib.import_module(f'strategies.{strategy_name}')
        strategy_class = getattr(module, 'Strategy')
        logger.info(f"加载外部策略: {strategy_name}")
        return strategy_class(params)
    except (ImportError, AttributeError) as e:
        logger.error(f"加载策略失败: {strategy_name}, 错误: {e}")
        available = list(BUILTIN_STRATEGIES.keys())
        raise ValueError(f"策略 '{strategy_name}' 不存在。可用策略: {available}")


def create_strategy_from_config(config: dict) -> BaseStrategy:
    """
    从配置字典创建策略

    参数:
        config: 策略配置字典，应包含:
            - name: 策略名称
            - params: 策略参数 (可选)

    返回值:
        BaseStrategy: 策略实例

    示例:
        config = {
            'name': 'moving_average',
            'params': {
                'ma_windows': [20, 60],
                'volume_ratio': 1.5
            }
        }
        strategy = create_strategy_from_config(config)
    """
    name = config.get('name')
    params = config.get('params', {})

    if not name:
        raise ValueError("配置必须包含 'name' 字段")

    return load_strategy(name, params)


__all__ = [
    'BaseStrategy',
    'BlackHorseStrategy',
    'MomentumReversal13Strategy',
    'MovingAverageStrategy',
    'load_strategy',
    'create_strategy_from_config',
    'list_strategies',
]
