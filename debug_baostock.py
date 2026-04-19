"""调试baostock API返回值"""

import baostock as bs
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_baostock_apis():
    """测试baostock各种API的返回值"""

    # 登录baostock
    lg = bs.login()
    logger.info(f"baostock登录: {lg.error_msg}")

    if lg.error_code != "0":
        logger.error("baostock登录失败")
        return

    try:
        # 测试几个股票代码
        test_symbols = ["sh.603123", "sz.000628", "sh.688002"]

        for symbol in test_symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试股票: {symbol}")
            logger.info(f"{'='*60}")

            # 1. 测试 query_stock_basic
            logger.info("\n1. query_stock_basic 返回结果:")
            rs = bs.query_stock_basic(code=symbol)
            if rs.error_code == "0":
                logger.info(f"字段: {rs.fields}")
                while rs.next():
                    data = rs.get_row_data()
                    logger.info(f"数据: {data}")
            else:
                logger.error(f"错误: {rs.error_msg}")

            # 2. 测试 query_stock_industry
            logger.info("\n2. query_stock_industry 返回结果:")
            rs = bs.query_stock_industry(code=symbol)
            if rs.error_code == "0":
                logger.info(f"字段: {rs.fields}")
                while rs.next():
                    data = rs.get_row_data()
                    logger.info(f"数据: {data}")
            else:
                logger.error(f"错误: {rs.error_msg}")

            # 3. 测试 query_history_k_data_plus 获取总股本
            logger.info("\n3. query_history_k_data_plus 返回结果:")
            today = datetime.now().strftime("%Y-%m-%d")
            # 先获取可用的字段
            rs_test = bs.query_history_k_data_plus(
                symbol,
                "date,code,open,high,low,close,volume,turn,tradestatus",
                start_date=today,
                end_date=today,
                frequency="d",
            )
            if rs_test.error_code == "0":
                logger.info(f"可用字段: {rs_test.fields}")

                # 尝试获取总股本
                rs_shares = bs.query_history_k_data_plus(
                    symbol,
                    "date,close,totalShares,totalShare",
                    start_date=today,
                    end_date=today,
                    frequency="d",
                )
                if rs_shares.error_code == "0":
                    logger.info(f"总股本查询字段: {rs_shares.fields}")
                    while rs_shares.next():
                        data = rs_shares.get_row_data()
                        logger.info(f"总股本数据: {data}")
                else:
                    logger.error(f"总股本查询错误: {rs_shares.error_msg}")
            else:
                logger.error(f"测试查询错误: {rs_test.error_msg}")

            # 4. 测试 query_profit_data 获取财务数据
            logger.info("\n4. query_profit_data 返回结果:")
            current_year = datetime.now().year
            rs_profit = bs.query_profit_data(
                code=symbol,
                year=current_year,
                quarter=4
            )
            if rs_profit.error_code == "0":
                logger.info(f"字段: {rs_profit.fields}")
                while rs_profit.next():
                    data = rs_profit.get_row_data()
                    logger.info(f"数据: {data}")
            else:
                logger.error(f"错误: {rs_profit.error_msg}")

                # 尝试前一年
                rs_profit = bs.query_profit_data(
                    code=symbol,
                    year=current_year - 1,
                    quarter=4
                )
                if rs_profit.error_code == "0":
                    logger.info(f"前一年字段: {rs_profit.fields}")
                    while rs_profit.next():
                        data = rs_profit.get_row_data()
                        logger.info(f"数据: {data}")
                else:
                    logger.error(f"前一年错误: {rs_profit.error_msg}")

            # 5. 测试 query_valuation_data 获取估值数据
            logger.info("\n5. query_valuation_data 返回结果:")
            rs_val = bs.query_valuation_data(
                code=symbol,
                date=today
            )
            if rs_val.error_code == "0":
                logger.info(f"字段: {rs_val.fields}")
                while rs_val.next():
                    data = rs_val.get_row_data()
                    logger.info(f"数据: {data}")
            else:
                logger.error(f"错误: {rs_val.error_msg}")

            logger.info(f"\n{'='*60}\n")

    finally:
        bs.logout()
        logger.info("baostock已登出")


if __name__ == "__main__":
    test_baostock_apis()