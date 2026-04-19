"""详细探索baostock所有API的功能和返回字段"""

import baostock as bs
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_baostock_apis():
    """详细探索baostock所有API"""

    # 登录baostock
    lg = bs.login()
    logger.info(f"baostock登录: {lg.error_msg}")

    if lg.error_code != "0":
        logger.error("baostock登录失败")
        return

    try:
        # 测试股票代码
        test_symbol = "sh.603123"  # 翠微股份
        today = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year

        print("=" * 80)
        print("baostock API 详细功能探索")
        print("=" * 80)

        # 1. 股票基本信息
        print("\n1. query_stock_basic - 股票基本信息")
        print("-" * 60)
        rs = bs.query_stock_basic(code=test_symbol)
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 2. 股票行业信息
        print("\n2. query_stock_industry - 股票行业信息")
        print("-" * 60)
        rs = bs.query_stock_industry(code=test_symbol)
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 3. 历史K线数据
        print("\n3. query_history_k_data_plus - 历史K线数据")
        print("-" * 60)
        rs = bs.query_history_k_data_plus(
            test_symbol,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date="2026-04-01",
            end_date=today,
            frequency="d",
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        # 4. 利润表数据
        print("\n4. query_profit_data - 利润表数据")
        print("-" * 60)
        rs = bs.query_profit_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 5. 资产负债表数据
        print("\n5. query_balance_data - 资产负债表数据")
        print("-" * 60)
        rs = bs.query_balance_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 6. 现金流量表数据
        print("\n6. query_cash_flow_data - 现金流量表数据")
        print("-" * 60)
        rs = bs.query_cash_flow_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 7. 杜邦分析数据
        print("\n7. query_dupont_data - 杜邦分析数据")
        print("-" * 60)
        rs = bs.query_dupont_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 8. 成长能力数据
        print("\n8. query_growth_data - 成长能力数据")
        print("-" * 60)
        rs = bs.query_growth_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 9. 运营能力数据
        print("\n9. query_operation_data - 运营能力数据")
        print("-" * 60)
        rs = bs.query_operation_data(
            code=test_symbol,
            year=current_year - 1,
            quarter=4
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 10. 除权除息信息
        print("\n10. query_dividend_data - 除权除息信息")
        print("-" * 60)
        rs = bs.query_dividend_data(
            code=test_symbol,
            year=current_year - 1
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 11. 复权因子
        print("\n11. query_adjust_factor - 复权因子")
        print("-" * 60)
        rs = bs.query_adjust_factor(
            code=test_symbol,
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        # 12. 业绩快报
        print("\n12. query_performance_express_report - 业绩快报")
        print("-" * 60)
        rs = bs.query_performance_express_report(
            code=test_symbol,
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 13. 业绩预告
        print("\n13. query_forecast_report - 业绩预告")
        print("-" * 60)
        rs = bs.query_forecast_report(
            code=test_symbol,
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            while rs.next():
                data = rs.get_row_data()
                print(f"数据: {data}")
        else:
            print(f"错误: {rs.error_msg}")

        # 14. 宏观经济数据 - 存款利率
        print("\n14. query_deposit_rate_data - 存款利率数据")
        print("-" * 60)
        rs = bs.query_deposit_rate_data(
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        # 15. 宏观经济数据 - 贷款利率
        print("\n15. query_loan_rate_data - 贷款利率数据")
        print("-" * 60)
        rs = bs.query_loan_rate_data(
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        # 16. 宏观经济数据 - 货币供应量
        print("\n16. query_money_supply_data_month - 货币供应量(月度)")
        print("-" * 60)
        rs = bs.query_money_supply_data_month(
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        # 17. 宏观经济数据 - 存款准备金率
        print("\n17. query_required_reserve_ratio_data - 存款准备金率数据")
        print("-" * 60)
        rs = bs.query_required_reserve_ratio_data(
            start_date="2026-01-01",
            end_date=today
        )
        if rs.error_code == "0":
            print(f"字段: {rs.fields}")
            count = 0
            while rs.next() and count < 3:
                data = rs.get_row_data()
                print(f"数据: {data}")
                count += 1
            if count == 3:
                print("... (只显示前3条)")
        else:
            print(f"错误: {rs.error_msg}")

        print("\n" + "=" * 80)
        print("API探索完成")
        print("=" * 80)

    finally:
        bs.logout()
        logger.info("baostock已登出")

if __name__ == "__main__":
    explore_baostock_apis()