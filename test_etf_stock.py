import baostock as bs
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_etf_and_stocks():
    """测试baostock是否包含ETF和个股信息"""

    # 登录baostock
    lg = bs.login()
    logger.info(f"baostock登录: {lg.error_msg}")

    if lg.error_code != "0":
        logger.error("baostock登录失败")
        return

    try:
        # 测试不同类型的代码
        test_codes = [
            # ETF代码（从4月15日文件中看到的）
            "sh.513520",  # ETF
            "sh.516010",  # ETF
            "sz.159100",  # ETF
            "sz.159110",  # ETF
            "sz.159363",  # ETF
            # 个股代码
            "sh.600673",  # 东阳光（个股）
            "sh.603156",  # 养元饮品（个股）
            "sz.300285",  # 国瓷材料（个股）
        ]

        for code in test_codes:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试代码: {code}")
            logger.info(f"{'='*60}")

            # 1. 测试 query_stock_basic 获取基本信息
            rs = bs.query_stock_basic(code=code)
            if rs.error_code == "0":
                logger.info("query_stock_basic 成功")
                while rs.next():
                    data = rs.get_row_data()
                    # 字段: ['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status']
                    code_name = data[1] if len(data) > 1 else "N/A"
                    stock_type = data[4] if len(data) > 4 else "N/A"
                    logger.info(f"股票名称: {code_name}")
                    logger.info(f"股票类型: {stock_type}")
                    logger.info(f"完整数据: {data}")
            else:
                logger.error(f"query_stock_basic 错误: {rs.error_msg}")

            # 2. 测试 query_history_k_data_plus 获取K线数据
            rs_k = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume",
                start_date="2026-04-20",
                end_date="2026-04-21",
                frequency="d",
            )
            if rs_k.error_code == "0":
                logger.info("query_history_k_data_plus 成功")
                count = 0
                while rs_k.next():
                    data = rs_k.get_row_data()
                    count += 1
                logger.info(f"获取到 {count} 条K线数据")
            else:
                logger.error(f"query_history_k_data_plus 错误: {rs.error_msg}")

            # 3. 测试 query_stock_industry 获取行业信息（ETF可能没有）
            rs_industry = bs.query_stock_industry(code=code)
            if rs_industry.error_code == "0":
                logger.info("query_stock_industry 成功")
                has_data = False
                while rs_industry.next():
                    data = rs_industry.get_row_data()
                    logger.info(f"行业信息: {data}")
                    has_data = True
                if not has_data:
                    logger.info("无行业信息（可能是ETF）")
            else:
                logger.info(f"query_stock_industry 错误或无数据: {rs_industry.error_msg}")

    finally:
        bs.logout()
        logger.info("baostock已登出")

if __name__ == "__main__":
    test_etf_and_stocks()