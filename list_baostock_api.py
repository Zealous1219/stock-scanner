import baostock as bs
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_baostock_apis():
    """列出baostock所有可用API"""
    # 股票指数成分股相关
    stock_apis = [
        # 成分股
        ("query_hs300_stocks", "沪深300成分股"),
        ("query_zz500_stocks", "中证500成分股"),
        # ("query_a500_stocks", "中证A500成分股"),  # 不存在
        # ("query_zz50_stocks", "中证50成分股"),    # 不存在

        # 全部股票
        ("query_all_stock", "所有股票列表(需指定日期)"),
    ]

    # 行情数据相关
    quote_apis = [
        ("query_history_k_data_plus", "历史K线数据"),
        ("query_trade_dates", "交易日历"),
        ("query_stock_basic", "股票基本信息"),
    ]

    # 财务数据相关
    finance_apis = [
        ("query_profit_sheet", "利润表"),
        ("query_balance_sheet", "资产负债表"),
        ("query_cash_flow_sheet", "现金流量表"),
        ("query_dupont_analysis", "杜邦分析"),
        ("query_growth_analysis", "成长能力分析"),
        ("query_profit_analysis", "盈利能力分析"),
        ("query_operation_analysis", "运营能力分析"),
        ("query_debtpaying_analysis", "偿债能力分析"),
        ("query_basic_news", "财经新闻"),
        ("query_industry_news", "行业新闻"),
        ("query_stock_news", "个股新闻"),
    ]

    print("=" * 60)
    print("baostock API 列表")
    print("=" * 60)

    print("\n【成分股相关】")
    for name, desc in stock_apis:
        func = getattr(bs, name, None)
        status = "[可用]" if func else "[不可用]"
        print(f"  {status}  {name:30s} - {desc}")

    print("\n【行情数据相关】")
    for name, desc in quote_apis:
        func = getattr(bs, name, None)
        status = "[可用]" if func else "[不可用]"
        print(f"  {status}  {name:30s} - {desc}")

    print("\n【财务数据相关】")
    for name, desc in finance_apis:
        func = getattr(bs, name, None)
        status = "[可用]" if func else "[不可用]"
        print(f"  {status}  {name:30s} - {desc}")

    # 尝试获取所有以query开头的函数
    print("\n【所有query_开头的函数】")
    query_funcs = [attr for attr in dir(bs) if attr.startswith('query_')]
    for func_name in sorted(query_funcs):
        print(f"  - {func_name}")

if __name__ == "__main__":
    list_baostock_apis()
