# 测试套件

## 运行

```bash
python -m pytest tests/ -v
```

## 测试文件说明

| 文件 | 覆盖范围 |
|------|----------|
| `test_data_utils.py` | 周线完成判定、交易日历查询、snapshot 交易周元数据 |
| `test_scanner_app.py` | 缓存刷新逻辑、forward returns、replay strategy slug |
| `test_strategy_semantics.py` | MR13 pivot anchor 三分支 + 5 种失败条件；BlackHorse 5 种场景 |
| `test_replay_observability.py` | weekly cache 等价性、completeness guard、切片语义 |
| `test_replay_per_snapshot.py` | per-snapshot 文件命名、覆盖写入、checkpoint 时序 |
| `test_replay_data_end_date.py` | replay 数据边界固定、checkpoint resume 校验 |
| `test_replay_strategy_isolation.py` | 多策略文件隔离、fresh/resume 防污染 |
| `test_replay_failed_symbols.py` | checkpoint 失败 symbol 记录 |

## 设计原则

- 不依赖真实 baostock 网络连接（mock/patch）
- 不污染 `data/`、`output/`、`config.json`
- 优先覆盖核心业务逻辑和边界条件
- 测试应专注于代码行为的可观察结果，而非实现细节
