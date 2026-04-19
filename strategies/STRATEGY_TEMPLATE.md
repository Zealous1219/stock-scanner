# 策略模块开发模板

本文档提供创建新策略模块的指南。

## 快速开始

创建一个新的策略模块，只需两步：

1. 在 `strategies/` 目录下创建新的Python文件
2. 继承 `BaseStrategy` 类并实现必要方法

## 模板代码

```python
"""
策略名称

策略说明:
    详细描述策略的原理和逻辑

参数:
    param1 (type): 参数说明
    param2 (type): 参数说明

使用示例:
    from strategies import load_strategy
    strategy = load_strategy('your_strategy_name', {
        'param1': value1,
        'param2': value2
    })

作者: Your Name
日期: 2026-03-28
"""

from typing import Dict, Any, List
import pandas as pd
from strategies import BaseStrategy


class Strategy(BaseStrategy):
    """
    策略名称

    策略详细说明，包括策略原理、适用场景等。
    """

    DEFAULT_PARAMS = {
        'param1': default_value1,
        'param2': default_value2,
    }

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

    def _init_strategy(self):
        """策略特定初始化"""
        self.param1 = self.params.get('param1', default_value1)
        self.param2 = self.params.get('param2', default_value2)

    def _validate_params(self):
        """参数验证"""
        if self.param1 < 0:
            raise ValueError("param1 must be >= 0")

        if self.param2 not in ['option1', 'option2']:
            raise ValueError("param2 must be 'option1' or 'option2'")

    def compute(self, df: pd.DataFrame) -> bool:
        """
        计算策略信号

        参数:
            df: 股票历史数据

        返回值:
            True表示符合策略条件
        """
        # 策略逻辑实现

        return True/False

    @property
    def supported_timeframes(self) -> List[str]:
        """支持的时间周期"""
        return ['daily', 'weekly']
```

## 必须实现的方法

| 方法 | 说明 | 必须实现 |
|------|------|----------|
| `_init_strategy()` | 策略初始化 | ✅ |
| `_validate_params()` | 参数验证 | ✅ |
| `compute()` | 计算策略信号 | ✅ |

## 可选重写的方法

| 方法 | 说明 | 默认行为 |
|------|------|----------|
| `get_params()` | 获取参数 | 返回params副本 |
| `set_params()` | 设置参数 | 更新params并重新初始化 |
| `get_name()` | 获取策略名 | 返回类名 |
| `get_description()` | 获取描述 | 返回类文档 |
| `get_required_columns()` | 所需数据列 | 返回基础列 |
| `support_timeframe()` | 支持的时间周期 | 检查是否在列表中 |
| `supported_timeframes` | 支持的时间周期列表 | ['daily', 'weekly'] |

## 使用策略

### 方式1: 动态加载

```python
from strategies import load_strategy

strategy = load_strategy('strategy_name', {
    'param1': value1
})
```

### 方式2: 直接导入

```python
from strategies.strategy_name import Strategy

strategy = Strategy({
    'param1': value1
})
```

### 方式3: 从配置加载

```python
from strategies import create_strategy_from_config

config = {
    'name': 'strategy_name',
    'params': {
        'param1': value1
    }
}
strategy = create_strategy_from_config(config)
```

## 策略注册

新策略创建后，会自动被 `load_strategy()` 和 `list_strategies()` 发现。

如果需要显式注册，可在 `strategies/__init__.py` 的 `BUILTIN_STRATEGIES` 字典中添加。
