# 策略开发模板

## 快速开始

1. 在 `strategies/` 下新建 `.py` 文件
2. 继承 `BaseStrategy` 实现 `scan()` 方法
3. 非必要无需注册——自动发现会加载 `strategies/` 下所有非 `_` 和 `base` 开头的模块。如需简短别名，在 `__init__.py` 的 `BUILTIN_STRATEGIES` 中添加

## 核心接口

| 方法 | 必须? | 说明 |
|------|:---:|------|
| `_init_strategy()` | **推荐** | 从 `self.params` 提取并初始化策略属性 |
| `_validate_params()` | **推荐** | 参数合法性校验，无效时抛 `ValueError` |
| `can_run(context)` | 可选 | 任务级执行判断，默认始终允许。返回 `StrategyDecision` |
| `scan(symbol, df, context)` | **必须** | 单标的完整扫描，返回 `StrategyResult` |
| `compute(df)` | 历史 | `compute()` 是历史兼容接口。默认 `scan()` 会调用它，但新策略直接实现 `scan()` 即可，无需实现 `compute()` |

**初始化链**：`DEFAULT_PARAMS` → 用户传入的 `params` → `_init_strategy()` → `_validate_params()`

**调用路径**：

```
scanner / replay  →  can_run(context)  →  StrategyDecision
                  →  scan(symbol, df, context, precomputed_weekly=None)  →  StrategyResult
```

## precomputed_weekly 参数（周线策略专用）

`scan()` 可接受额外参数 `precomputed_weekly: pd.DataFrame | None = None`。

- **scanner** 始终传 `None`（策略走标准路径调用 `get_completed_weekly_bars`）
- **replay** 框架传入预切片周线 DataFrame 以跳过重复 reample（仅 `black_horse` 和 `momentum_reversal_13` 会收到非 None 值）

周线策略的标准处理模式：

```python
def scan(self, symbol, df, context, precomputed_weekly=None) -> StrategyResult:
    if precomputed_weekly is not None:
        weekly = precomputed_weekly  # replay 加速路径
    else:
        weekly = get_completed_weekly_bars(df, now=context.now)  # scanner 标准路径
    ...
```

非周线策略（如 `moving_average`）不需要此参数。

## details 结构要求

`StrategyResult.details` 中的 dict 应包含以下键以确保 replay 输出完整：

| 键 | 必要? | 说明 |
|----|:---:|------|
| `symbol` | 必须 | 股票代码，用于输出记录和错误跟踪 |
| `signal_date` | 信号时 | 信号确认日期（`YYYY-MM-DD` 字符串），用于计算 forward returns |
| `signal_type` | 推荐 | 信号分类标识（如 `"black_horse_ready"`, `"momentum_reversal_13"`） |

## 示例框架

```python
"""策略名称 — 一句话说明"""

from strategy_runtime import StrategyContext, StrategyDecision, StrategyResult
from strategies.base import BaseStrategy


class Strategy(BaseStrategy):
    DEFAULT_PARAMS = {"param": 10}

    def _init_strategy(self):
        self.param = int(self.params.get("param", 10))

    def _validate_params(self):
        if self.param < 1:
            raise ValueError("param 必须 >= 1")

    def can_run(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(should_run=True, reason_code="ok", reason_text="")

    def scan(self, symbol, df, context, precomputed_weekly=None) -> StrategyResult:
        # 如果是周线策略，先处理 precomputed_weekly
        # weekly = precomputed_weekly if precomputed_weekly is not None else get_completed_weekly_bars(df, now=context.now)

        # 实现策略逻辑
        ...

        return StrategyResult(
            matched=True,
            reason_code="matched",
            reason_text="信号确认",
            details={"symbol": symbol, "signal_type": "my_signal", "signal_date": "2025-01-10"},
        )

    @property
    def supported_timeframes(self) -> list:
        return ["weekly", "daily"]  # 周线策略；日线策略反过来
```

## 注意事项

- `reason_code` 推荐 `snake_case`，每个失败条件有独立 code 便于分析
- `compute()` 不会反向调用 `scan()`，两个路径独立
- 参考已有实现：`black_horse.py`（简洁，3 个条件）和 `momentum_reversal_13.py`（复杂，多分支搜索）
- 模块末尾需导出：`Strategy = YourStrategyClass`
