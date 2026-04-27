# 股票扫描器测试套件

本目录包含股票扫描器项目的正式测试用例，专注于验证核心业务逻辑的正确性。

## 测试文件结构

```
tests/
├── __init__.py          # 测试包初始化
├── test_data_utils.py   # 数据工具函数测试
├── test_scanner_app.py  # 扫描器应用逻辑测试
└── README.md           # 本文件
```

## 运行测试

### 基本命令

```bash
# 在项目根目录运行
python -m pytest tests/ -v

# 或使用简写
pytest tests/ -v
```

### 运行特定测试文件

```bash
# 运行 scanner_app 的所有测试
python -m pytest tests/test_scanner_app.py -v

# 运行 data_utils 的所有测试  
python -m pytest tests/test_data_utils.py -v
```

### 测试项目详情

### test_data_utils.py

测试 `data_utils.py` 中的周线完成判定逻辑。

#### TestGetWeekMondayFriday
- `test_wednesday_input`: 周三输入时，周一/周五计算正确
- `test_saturday_input`: 周六输入时，不会漂到下周五

#### TestGetLastTradingDayOfWeek
- `test_normal_trading_week_returns_friday`: 正常交易周返回周五
- `test_short_week_returns_last_trading_day`: 短周返回本周最后交易日（如周四）
- `test_no_trading_days_in_week_returns_none`: 本周没有交易日时返回 None
- `test_query_failure_returns_none`: 查询失败时返回 None
- `test_exception_returns_none`: baostock调用异常时返回 None

#### TestGetLastCompletedWeekEnd
测试 `get_last_completed_week_end` 函数，覆盖8个关键场景：
- 周五19:59、周五20:01、周六等各种场景
- 包含边界条件和异常处理

### test_scanner_app.py

测试 `scanner_app.py` 中的缓存刷新逻辑和股票池回退逻辑。

#### TestShouldForceRefreshOnFriday
- `test_friday_before_2000_returns_false`: 周五20:00前返回False
- `test_friday_after_2000_returns_true`: 周五20:00后返回True
- `test_non_friday_returns_false`: 非周五返回False

#### TestLoadOrUpdateData
- `test_friday_2030_with_latest_date_today_uses_cache`: 周五20:30，latest_date已是今天不调用fetch
- `test_friday_2030_with_latest_date_yesterday_fetches_new_data`: 周五20:30，latest_date小于今天调用fetch
- `test_fetch_historical_data_called_with_string_dates`: 验证参数是字符串不是Timestamp
- `test_file_not_exists_fetches_new_data`: 文件不存在时获取新数据

#### TestGetLatestTradingDay
- `test_returns_latest_trading_day`: 能返回最近交易日
- `test_query_failure_raises_exception`: 查询失败时抛出异常
- `test_no_trading_days_raises_exception`: 没有交易日时抛出异常

#### TestGetStockList
- `test_all_pool_uses_latest_trading_day`: get_stock_list("all")会使用get_latest_trading_day
- `test_hs300_pool_does_not_use_trading_day`: hs300股票池不使用trading day
- `test_empty_stock_list_raises_exception`: 空股票列表抛出异常
- `test_unsupported_pool_raises_exception`: 不支持的股票池抛出异常

## 测试设计原则

- **独立性**：测试不依赖真实 baostock 登录或网络请求
- **安全性**：不污染 data/、output/、config.json
- **覆盖率**：优先覆盖核心业务逻辑和边界条件
- **Mock 策略**：使用 mock/patch 控制外部依赖

## 添加新测试

遵循现有模式添加测试：
1. 在相应测试文件中添加新的测试类或方法
2. 使用相同的 mock 模式
3. 运行新添加的测试