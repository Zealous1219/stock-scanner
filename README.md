# Stock Scanner

A-share 选股扫描器，支持多策略并行扫描和历史 replay 验证。

---

## 架构概览

```
scanner_app.py     —— 主入口（main）及 replay 验证框架
data_utils.py      —— 周线转换、交易日历查询、周线完成判定
strategy_runtime.py—— 运行时数据结构（Context / Decision / Result）
strategies/        —— 策略模块（black_horse, momentum_reversal_13, moving_average）
config_loader.py   —— JSON 配置加载
stock-scanner.py   —— 薄启动器，用于本地覆盖
```

**分层原则**：data（数据获取/缓存）、strategy（策略逻辑）、backtest（回测框架）、output（输出）保持独立。

---

## 快速开始

### 环境检查与安装

```powershell
.\install_requirements.ps1   # 安装依赖
.\check_runtime_env.ps1      # 检查运行环境
```

### 运行 scanner

编辑 `config.json` 选择策略和股票池，然后：

```powershell
.\run_scanner.ps1
# 或直接
py -3.13 stock-scanner.py
```

### 运行 replay 验证

```python
py -3.13 -c "from scanner_app import run_weekly_replay_validation; run_weekly_replay_validation()"
```

### 运行测试

```powershell
python -m pytest tests/ -v
```

---

## 配置

### config.json

```json
{
  "strategy": {
    "name": "black_horse",
    "params": { "required_weeks": 3, "min_weekly_bars": 12 }
  },
  "stock_pool": { "type": "hs300" },
  "data": { "lookback_days": 180, "initial_days": 400, "request_interval": 0.5 }
}
```

### 配置优先级（高 → 低）

1. **环境变量**：`STOCK_POOL`、`STRATEGY_NAME`
2. **`stock-scanner.py` 本地常量**
3. **`config.json`**

示例：`set STOCK_POOL=zz500` 会覆盖 `config.json` 中的设置。

---

## 内置策略

| 策略名 | 说明 | 配置 name | 默认参数 |
|--------|------|-----------|----------|
| **Black Horse** | 检测连续三周阳线 + 涨幅递增 + 成交量递增 | `black_horse` / `bh` | `required_weeks=3`, `min_weekly_bars=12` |
| **动量反转13周** | 检测 10 周下跌后的动量反转信号（pivot anchor 模型定位 big1） | `momentum_reversal_13` / `mr13` | `min_downtrend_weeks=10`, `min_weekly_bars=24`, `reversal_weeks=3` |
| **移动均线** | 均线交叉/成交量放大检测 | `moving_average` / `ma` | `ma_windows=[20,60]`, `volume_ratio=1.5` |

开发新策略请参考 [strategies/STRATEGY_TEMPLATE.md](strategies/STRATEGY_TEMPLATE.md)。

### 切换策略

编辑 `config.json` 的 `strategy.name`，或设置 `STRATEGY_NAME` 环境变量。后者会自动从目标策略类读取 `DEFAULT_PARAMS` 作为参数。

### 股票池

`hs300` / `zz500` / `sz50` / `all` （全 A 股需指定交易日）

---

## 数据

| 项目 | 说明 |
|------|------|
| 数据源 | baostock（A 股日线 OHLCV） |
| 缓存目录 | `data/`（按 symbol 存储 CSV，如 `data/sh_600000.csv`） |
| 刷新策略 | 周五 20:00 后强制刷新当日数据；其余时间若缓存最后日期 ≥ 昨日则直接使用 |
| 周线生成 | `W-FRI` reample：将日线按周五锚点聚合为周线，包含 `trading_days_count` 和 `last_daily_date` 元字段 |
| 周线完成判定 | `get_last_completed_week_end` 结合当前时间、最新日线日期、交易日历共同判断本周是否完成 |

---

## Scanner 输出

### output/

| 文件 | 说明 |
|------|------|
| `<strategy>_candidates_YYYY-MM-DD.csv` | 策略命中的候选股票列表 |
| `strategy_runs_YYYY-MM-DD.csv` | 本次运行日志（执行状态、统计、配置快照） |

---

## Replay 验证

### 概念

Replay 是对历史时间点进行策略回放：取过去 52 周（默认）的每个已完成周作为 **snapshot**（snapshot_date = 周五 23:59:59），加载该时间点可见的数据，运行策略逻辑，记录匹配结果并计算前向收益率（4w/8w/12w/16w/20w）。目的是验证策略在真实历史环境中的信号质量和可复现性。

### Snapshot 生成

`generate_weekly_snapshot_dates(52)` 从最近一个已完成的周五开始向前递减 52 周。当前未完成周不包含在内。每个 snapshot 的 hour=23:59:59 保证 `get_last_completed_week_end` 将本周判为"已完成"。

### 运行模式

- **Fresh**：从头运行全部 52 个 snapshot。如有同一策略的旧输出文件但无 checkpoint，拒绝运行以防止静默覆盖。手动删除旧文件后方可开始。
- **Resume**（默认）：检测到 checkpoint 后从上次中断处续跑。checkpoint 记录已完成 snapshot 列表、snapshot_dates 窗口身份、replay_data_end_date 数据边界。三者均通过校验才允许 resume。

```python
run_weekly_replay_validation(resume=True)   # resume（默认）
run_weekly_replay_validation(resume=False)  # fresh
```

### Checkpoint 机制

Checkpoint 是 `validation/replay/replay_{strategy_slug}_...checkpoint.json`，记录：

| 字段 | 作用 |
|------|------|
| `completed_snapshots` | 已完成的 snapshot 日期列表 |
| `snapshot_dates` | 完整 52 个 snapshot 窗口身份。resume 时与当前运行比对，不一致则拒绝 |
| `replay_data_end_date` | 固定数据边界（`last_snapshot + 21 周`）。resume 时比对，保证数据一致性 |
| `failed_symbols_per_snapshot` | （可选）失败 symbol 记录，仅做质量观察不阻断 resume |

### 关键特性

- **策略文件隔离**：`mr13` 和 `black_horse` 的 checkpoint/结果/错误文件互相独立
- **Per-snapshot 输出**：每个 snapshot 独立 CSV，覆盖写入（非追加），崩溃重跑不产生重复行
- **Weekly cache 优化**：周线策略预计算全量周线一次，每个 snapshot 仅做廉价切片
- **符号级守卫**：`_guard_snapshot_week_completion` 在 snapshot 边界校验末根周线数据完整性，防止半周 phantom bar

### 输出目录

```
validation/replay/
├── replay_mr13_all_52w_v1_2025-01-10.csv       # snapshot 命中结果
├── replay_mr13_all_52w_v1_2025-01-10_errors.csv # snapshot 错误记录
├── replay_mr13_all_52w_v1.checkpoint.json       # 进度检查点
├── replay_black_horse_all_52w_v1_2025-01-10.csv
└── ...
```

### 合并结果

```python
from scanner_app import merge_replay_snapshot_files
merge_replay_snapshot_files("mr13", "all", 52, "v1")
# 输出: validation/replay/replay_mr13_all_52w_v1_merged.csv
```

---

## 项目结构

```
├── scanner_app.py            主入口 & replay 框架
├── data_utils.py             周线工具函数
├── strategy_runtime.py       运行时数据结构
├── config_loader.py          配置加载
├── stock-scanner.py          薄启动器
├── config.json               配置文件
├── AGENTS.md                 AI agent 工作规则
├── claude.md                 项目上下文（供 AI agent 使用）
├── strategies/
│   ├── __init__.py           策略注册/加载
│   ├── base.py               策略基类
│   ├── black_horse.py        黑马周线策略
│   ├── momentum_reversal_13.py 动量反转13周策略
│   ├── moving_average.py     移动均线策略
│   └── STRATEGY_TEMPLATE.md  新策略开发模板
├── data/                     日线数据缓存（gitignored）
├── output/                   scanner 运行输出（gitignored）
├── validation/               replay 运行输出（gitignored）
├── tests/                    测试套件
└── local_tools/              手动研究辅助工具（筛选/富化/生成报告，不属于主流水线）
```

---

## 作者

zealous
