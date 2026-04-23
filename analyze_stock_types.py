import baostock as bs
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_stock_types():
    """分析baostock中的股票类型"""

    print("=" * 80)
    print("baostock 股票类型分析")
    print("=" * 80)

    # 从4月15日文件中提取一些代码进行分析
    codes_from_0415 = [
        "sh.513520", "sh.516010",  # 疑似ETF
        "sz.159100", "sz.159110", "sz.159363",  # 疑似ETF
        "sh.603123", "sh.603317",  # 个股
        "sz.000628", "sz.002281",  # 个股
    ]

    # 从4月21日文件中提取代码
    codes_from_0421 = [
        "sh.600673", "sh.600959", "sh.600977", "sh.603156",
        "sh.688183", "sh.688322", "sz.300285", "sz.300395"
    ]

    print("\n1. 从4月15日文件中提取的代码分析:")
    print("-" * 60)

    # 分析代码模式
    etf_patterns = [
        ("sh.51", "上证ETF/LOF"),
        ("sh.56", "上证ETF"),
        ("sz.15", "深证ETF/LOF"),
        ("sz.16", "深证ETF"),
        ("sz.159", "深证ETF"),
    ]

    stock_patterns = [
        ("sh.60", "上证A股"),
        ("sh.68", "上证科创板"),
        ("sz.00", "深证主板"),
        ("sz.30", "深证创业板"),
        ("sz.002", "深证中小板"),
    ]

    for code in codes_from_0415:
        found = False
        for prefix, desc in etf_patterns:
            if code.startswith(prefix):
                print(f"{code:12} - {desc} (疑似ETF)")
                found = True
                break

        if not found:
            for prefix, desc in stock_patterns:
                if code.startswith(prefix):
                    print(f"{code:12} - {desc} (个股)")
                    found = True
                    break

        if not found:
            print(f"{code:12} - 未知类型")

    print("\n2. 从4月21日文件中提取的代码分析:")
    print("-" * 60)

    for code in codes_from_0421:
        found = False
        for prefix, desc in stock_patterns:
            if code.startswith(prefix):
                print(f"{code:12} - {desc} (个股)")
                found = True
                break

        if not found:
            print(f"{code:12} - 未知类型")

    print("\n3. baostock 代码前缀分析:")
    print("-" * 60)

    code_prefixes = {
        "sh.": "上海证券交易所",
        "sz.": "深圳证券交易所",
        "bj.": "北京证券交易所",
    }

    sub_prefixes = {
        "sh.60": "上证A股",
        "sh.68": "上证科创板",
        "sh.50": "上证指数/ETF",
        "sh.51": "上证ETF/LOF",
        "sh.56": "上证ETF",
        "sz.00": "深证主板",
        "sz.30": "深证创业板",
        "sz.002": "深证中小板",
        "sz.15": "深证ETF/LOF",
        "sz.16": "深证ETF",
        "sz.159": "深证ETF",
        "bj.": "北交所股票",
    }

    print("主要交易所前缀:")
    for prefix, desc in code_prefixes.items():
        print(f"  {prefix:6} - {desc}")

    print("\n详细子前缀:")
    for prefix, desc in sub_prefixes.items():
        print(f"  {prefix:6} - {desc}")

    print("\n4. 结论:")
    print("-" * 60)
    print("根据代码前缀分析：")
    print("1. baostock 包含 ETF 数据（代码前缀为 sh.50/sh.51/sh.56/sz.15/sz.16/sz.159）")
    print("2. baostock 也包含个股数据（代码前缀为 sh.60/sh.68/sz.00/sz.30/sz.002）")
    print("3. 4月15日文件中包含 ETF 和个股混合")
    print("4. 4月21日文件中只包含个股（全部是 sh.60/sh.68/sz.30 前缀）")
    print("5. 策略可能过滤掉了 ETF，或者市场条件变化导致 ETF 不再符合黑马条件")

if __name__ == "__main__":
    analyze_stock_types()