# 策略模块开发模板

本文档提供创建新策略模块的推荐写法。

## 中文版使用说明

### 核心概念

- 策略基类要求：`BaseStrategy` 提供了 `can_run()` 和 `scan()` 的默认实现，新策略可覆盖这些方法。
- 推荐运行时接口：扫描器运行时 `scanner_app.py` 调用 `can_run(context)` 和 `scan(symbol, df, context)`，这是新策略的主接口。
- `compute()` 的定位：历史辅助方法，基类 `scan()` 默认实现会调用它，但新策略直接实现 `scan()` 时不再需要。
- 结构化返回：推荐使用 `StrategyDecision` 和 `StrategyResult` 返回是否执行、是否命中、原因码和明细信息。

### 设计原则

1. `can_run()` 负责任务级判断，例如时间窗口、市场状态、运行条件。
2. `scan()` 负责单标的完整扫描，并返回结构化结果。
3. `compute()` 是历史辅助方法，新策略直接实现 `scan()` 时无需关心。
4. 新策略建议优先采用 `can_run() + scan()` 模式。
5. 现有以 `compute()` 为核心的历史策略仍可继续存在。

### 关键接口说明

- `can_run(context)`：决定本次任务是否应执行该策略，返回 `StrategyDecision`
- `scan(symbol, df, context)`：对单个标的执行完整扫描，返回 `StrategyResult`
- `compute(df)`：历史辅助方法，基类 `scan()` 默认实现会调用它；新策略直接实现 `scan()` 时可不实现
- `StrategyContext`：运行时上下文，包含时间、股票池、配置等信息
- `StrategyDecision`：任务级执行决策
- `StrategyResult`：单标的扫描结果

## 快速开始

创建一个新策略模块，只需两步：

1. 在 `strategies/` 目录下创建新的 Python 文件
2. 继承 `BaseStrategy` 并实现必要方法

## 模板代码

```python
"""
策略名称

策略说明:
    详细描述策略原理、使用场景和主要判断条件

参数:
    param1 (type): 参数说明
    param2 (type): 参数说明

使用示例:
    from strategies import load_strategy
    strategy = load_strategy('your_strategy_name', {
        'param1': value1,
        'param2': value2
    })

"""

from typing import Dict, Any, List

import pandas as pd

from strategies import BaseStrategy
from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult


class Strategy(BaseStrategy):
    """
    策略名称

    详细描述策略原理、适用场景等。
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

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        """
        推荐运行时接口：任务级执行判断
        """
        return StrategyDecision(
            should_run=True,
            reason_code="always_run",
            reason_text="策略可以执行"
        )

    def scan(self, symbol: str, df: pd.DataFrame, context: StrategyContext) -> StrategyResult:
        """
        推荐运行时接口：单标的完整扫描
        
        说明：
        - 应在此方法中完成所有策略逻辑组装
        - 复杂信号或数值计算应在此处理
        - 可调用 compute() 获取简单布尔信号，但不是必须
        """
        try:
            if len(df) < 20:
                return StrategyResult(
                    matched=False,
                    reason_code="insufficient_data",
                    reason_text=f"股票 {symbol} 数据不足，无法执行策略判断",
                    details={"symbol": symbol, "available_bars": int(len(df))}
                )

            latest = df.iloc[-1]

            # 在此处实现具体策略逻辑
            # 方式1：直接在 scan() 中完成所有逻辑（推荐）
            # 方式2：调用 self.compute(df) 获取布尔信号
            
            # 示例：简单的价格突破判断
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            matched = latest["close"] > latest["open"] and latest["volume"] > prev["volume"]

            if matched:
                return StrategyResult(
                    matched=True,
                    reason_code="strategy_matched",
                    reason_text=f"股票 {symbol} 符合策略条件",
                    details={
                        "symbol": symbol,
                        "signal_type": "your_signal_type",
                        "latest_date": str(pd.to_datetime(latest["date"]).date()),
                        "close": float(latest["close"]),
                    },
                )

            return StrategyResult(
                matched=False,
                reason_code="strategy_not_matched",
                reason_text=f"股票 {symbol} 不符合策略条件",
                details={"symbol": symbol},
            )

        except Exception as e:
            return StrategyResult(
                matched=False,
                reason_code="computation_error",
                reason_text=f"策略计算错误: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )

    def compute(self, df: pd.DataFrame) -> bool:
        """
        历史辅助方法，基类 scan() 默认实现会调用此方法。

        新策略如果直接实现了 scan()，则无需实现此方法。
        此方法不会反向调用 scan()，兼容路径是单向的。
        """
        raise NotImplementedError("需要实现 compute() 或直接实现 scan()")

    @property
    def supported_timeframes(self) -> List[str]:
        return ['daily', 'weekly']
```

## 方法要求说明

### 推荐运行时接口（新策略主合约）

| 方法 | 说明 | 推荐实现 | 备注 |
|------|------|----------|------|
| `can_run()` | 运行时决策 | 是 | 扫描器运行时调用的入口；基类默认始终允许执行 |
| `scan()` | 单标的扫描 | 是 | 扫描器运行时调用的入口；基类默认调用 compute() |

### 历史辅助方法

| 方法 | 说明 | 需要实现 | 备注 |
|------|------|----------|------|
| `compute()` | 布尔策略信号 | 否 | 基类 scan() 默认实现会调用；新策略直接实现 scan() 时可不实现 |

### 可选重写（Optional override）

| 方法 | 说明 | 建议实现 | 备注 |
|------|------|----------|------|
| `_init_strategy()` | 策略初始化 | 是 | 策略特定初始化逻辑 |
| `_validate_params()` | 参数验证 | 是 | 确保参数有效 |

### 可选重写的方法

| 方法 | 说明 | 默认行为 |
|------|------|----------|
| `get_params()` | 获取参数 | 返回 `params` 副本 |
| `set_params()` | 设置参数 | 更新参数并重新初始化 |
| `supported_timeframes` | 支持的时间周期 | `['daily', 'weekly']` |

## 新策略开发推荐规范

### 推荐接口模式

新策略推荐采用以下模式：

1. `can_run(context: StrategyContext) -> StrategyDecision`
2. `scan(symbol: str, df: pd.DataFrame, context: StrategyContext) -> StrategyResult`

### `compute()` 的定位

 - **当前状态**：基类提供默认实现（raise NotImplementedError），不再强制要求
 - **兼容路径**：基类 `scan()` 默认实现会调用 `compute()`（单向，compute() 不会反向调用 scan()）
 - **新策略做法**：直接实现 `scan()`，`compute()` 不再需要
 - **已有策略做法**：只实现 `compute()` 即可通过默认 `scan()` 与运行时对接

### 设计建议

1. 在 `scan()` 中返回清晰的 `reason_code` 和 `reason_text`
2. 通过 `details` 返回策略特定信息，便于调试和分析
3. 在 `can_run()` 中实现时间条件或运行门禁
4. 保持策略实例尽量无状态，减少隐藏副作用

### 历史与推荐模式

- `MovingAverageStrategy` 更接近历史实现方式，可作为兼容参考
- `BlackHorseStrategy` 更接近当前推荐的 `can_run() + scan()` 模式

## 使用策略

### 方式1：动态加载

```python
from datetime import datetime

from strategies import load_strategy
from strategy_runtime import StrategyContext

strategy = load_strategy('strategy_name', {
    'param1': value1
})

context = StrategyContext(now=datetime.now(), stock_pool='all', config={})

decision = strategy.can_run(context)
if decision.should_run:
    result = strategy.scan('000001.SZ', df, context)
    if result.matched:
        print(f"匹配: {result.reason_text}")
        print(f"详情: {result.details}")
```

### 方式2：直接导入

```python
from strategies.strategy_name import Strategy

strategy = Strategy({
    'param1': value1
})
```

## 调试与测试建议

### 单元测试模板

```python
import pandas as pd
from datetime import datetime

from strategies.strategy_name import Strategy
from strategy_runtime import StrategyContext


def test_can_run():
    strategy = Strategy()
    context = StrategyContext(now=datetime.now(), stock_pool='all')

    decision = strategy.can_run(context)
    assert isinstance(decision.should_run, bool)
    assert isinstance(decision.reason_code, str)
    assert isinstance(decision.reason_text, str)


def test_scan():
    strategy = Strategy()
    context = StrategyContext(now=datetime.now(), stock_pool='all')

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'open': range(100),
        'high': range(100),
        'low': range(100),
        'close': range(100),
        'volume': range(100),
    })

    result = strategy.scan('000001.SZ', df, context)
    assert isinstance(result.matched, bool)
    assert isinstance(result.reason_code, str)
    assert isinstance(result.reason_text, str)
    assert isinstance(result.details, dict)
```

## 常见问题

1. **Q: 为什么推荐使用 `can_run()` 和 `scan()`？**  
   A: 因为这更贴近当前扫描器运行时的调用方式，也更适合返回结构化结果。

2. **Q: `compute()` 还需要实现吗？**  
   A: 取决于策略设计。如果直接实现了 `scan()`，则 `compute()` 不需要实现；如果只实现 `compute()`，则基类 `scan()` 默认会调用它。

3. **Q: 新策略应该以哪个接口为主？**  
   A: 建议以 `can_run() + scan()` 为主，把 `compute()` 视为内部辅助或历史兼容接口。

4. **Q: 现有策略需要马上迁移吗？**  
   A: 不需要。现有策略可以继续工作；本模板主要用于指导后续新增策略。

## 策略注册

新策略创建后，会自动被 `load_strategy()` 和 `list_strategies()` 发现。

如果需要显式注册，可在 `strategies/__init__.py` 的 `BUILTIN_STRATEGIES` 字典中添加。
