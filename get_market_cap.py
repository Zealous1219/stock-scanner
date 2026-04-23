"""获取股票市值信息"""

import baostock as bs
import pandas as pd
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_stock_market_cap(symbol: str) -> dict:
    """获取单只股票的市值信息"""
    try:
        # 使用 query_stock_basic 获取股票基本信息
        rs = bs.query_stock_basic(code=symbol)

        if rs.error_code != "0":
            logger.warning(f"获取 {symbol} 基本信息失败: {rs.error_msg}")
            return {"code": symbol, "market_cap": None, "pe_ratio": None, "pb_ratio": None, "name": None}

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            logger.warning(f"未找到 {symbol} 的基本信息")
            return {"code": symbol, "market_cap": None, "pe_ratio": None, "pb_ratio": None, "name": None}

        # 解析基本信息
        basic_info = data_list[0]
        code_name = basic_info[1] if len(basic_info) > 1 else None

        # 获取行业信息
        rs_industry = bs.query_stock_industry(code=symbol)
        industry_info = {}
        if rs_industry.error_code == "0":
            while rs_industry.next():
                industry_info = rs_industry.get_row_data()

        # 获取最新股价
        today = datetime.now().strftime("%Y-%m-%d")
        rs_k = bs.query_history_k_data_plus(
            symbol,
            "date,close",
            start_date=today,
            end_date=today,
            frequency="d",
        )

        close_price = None
        if rs_k.error_code == "0":
            k_data_list = []
            while rs_k.next():
                k_data_list.append(rs_k.get_row_data())

            if k_data_list:
                k_data = k_data_list[0]
                close_price = float(k_data[1]) if k_data[1] != '' else None

        # 获取财务数据（总股本和净利润）
        current_year = datetime.now().year
        market_cap = None
        pe_ratio = None
        total_share = None
        net_profit = None

        # 尝试获取最近4个季度的数据
        for year_offset in range(0, 2):  # 当前年和前一年
            for quarter in [4, 3, 2, 1]:  # 从Q4到Q1
                rs_profit = bs.query_profit_data(
                    code=symbol,
                    year=current_year - year_offset,
                    quarter=quarter
                )

                if rs_profit.error_code == "0":
                    profit_data_list = []
                    while rs_profit.next():
                        profit_data_list.append(rs_profit.get_row_data())

                    if profit_data_list:
                        profit_data = profit_data_list[0]
                        # 字段: ['code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin',
                        #        'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']
                        try:
                            # 总股本（单位：万股）
                            total_share_str = profit_data[9] if len(profit_data) > 9 else ''
                            net_profit_str = profit_data[6] if len(profit_data) > 6 else ''

                            if total_share_str and total_share_str != '':
                                total_share = float(total_share_str)  # 单位：万股

                            if net_profit_str and net_profit_str != '':
                                net_profit = float(net_profit_str)  # 单位：万元

                            if total_share is not None:
                                break  # 找到总股本数据就退出
                        except (ValueError, IndexError) as e:
                            logger.debug(f"解析财务数据时出错: {e}")
                            continue

            if total_share is not None:
                break

        # 计算市值（如果都有数据）
        if close_price is not None and total_share is not None:
            # 总股本单位是万股，需要转换为股：total_share * 10000
            # 市值 = 股价 * 总股本（单位：万元）
            market_cap = round(close_price * total_share * 10000 / 10000, 2)

        # 计算市盈率（如果都有数据）
        if market_cap is not None and net_profit is not None and net_profit > 0:
            # market_cap 单位是万元，net_profit 单位也是万元
            # 市盈率 = 市值（万元）/ 净利润（万元）
            pe_ratio = round(market_cap / net_profit, 2)

        # 获取市净率（PB）可能需要其他API，这里先留空
        pb_ratio = None

        return {
            "code": symbol,
            "name": code_name,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "industry": industry_info[3] if len(industry_info) > 3 else None if industry_info else None,
            "total_share": total_share,
            "close_price": close_price,
            "net_profit": net_profit
        }

    except Exception as e:
        logger.error(f"处理 {symbol} 时出错: {e}")
        return {"code": symbol, "market_cap": None, "pe_ratio": None, "pb_ratio": None, "name": None}


def main():
    """主函数：读取CSV文件并获取股票市值"""
    # 读取CSV文件
    csv_file = "output/black_horse_candidates_2026-04-21.csv"

    try:
        df = pd.read_csv(csv_file)
        logger.info(f"成功读取 {len(df)} 条记录")

        # 获取股票代码列表
        symbols = df["code"].tolist()

        # 登录baostock
        lg = bs.login()
        logger.info(f"baostock登录: {lg.error_msg}")

        if lg.error_code != "0":
            logger.error("baostock登录失败")
            return

        try:
            # 获取每只股票的市值信息
            results = []
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"处理 {i}/{len(symbols)}: {symbol}")

                market_cap_info = get_stock_market_cap(symbol)
                results.append(market_cap_info)

                # 避免请求过于频繁
                if i < len(symbols):
                    time.sleep(0.5)

            # 创建结果DataFrame
            result_df = pd.DataFrame(results)

            # 合并原始数据
            merged_df = pd.merge(df, result_df, on="code", how="left")

            # 保存结果
            output_file = "output/black_horse_candidates_with_market_cap_2026-04-21.csv"
            merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            logger.info(f"结果已保存到: {output_file}")

            # 打印摘要信息
            print("\n" + "="*80)
            print("股票市值信息摘要")
            print("="*80)

            # 按市值排序
            sorted_df = merged_df.sort_values("market_cap", ascending=False)

            print(f"\n总共有 {len(sorted_df)} 只股票")
            print(f"有市值信息的股票: {sorted_df['market_cap'].notna().sum()} 只")

            if sorted_df['market_cap'].notna().any():
                print(f"\n市值排名前10的股票:")
                print("-"*80)
                print(f"{'排名':<4} {'代码':<10} {'名称':<15} {'市值(元)':<20} {'市盈率':<10} {'行业':<20}")
                print("-"*80)

                for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
                    market_cap = f"{row['market_cap']:.2f}" if pd.notna(row['market_cap']) else "N/A"
                    pe_ratio = f"{row['pe_ratio']:.2f}" if pd.notna(row['pe_ratio']) else "N/A"
                    name = row['name'] if pd.notna(row['name']) else "N/A"
                    industry = row['industry'] if 'industry' in row and pd.notna(row['industry']) else "N/A"

                    print(f"{i:<4} {row['code']:<10} {name:<15} {market_cap:<12} {pe_ratio:<10} {industry:<20}")

            # 统计市值分布
            if sorted_df['market_cap'].notna().any():
                print(f"\n市值分布:")
                print("-"*40)

                # 定义市值区间
                bins = [0, 50, 100, 200, 500, 1000, float('inf')]
                labels = ['<50亿', '50-100亿', '100-200亿', '200-500亿', '500-1000亿', '>1000亿']

                sorted_df['market_cap_range'] = pd.cut(
                    sorted_df['market_cap'],
                    bins=bins,
                    labels=labels,
                    right=False
                )

                distribution = sorted_df['market_cap_range'].value_counts().sort_index()
                for range_label, count in distribution.items():
                    percentage = count / len(sorted_df) * 100
                    print(f"{range_label}: {count} 只 ({percentage:.1f}%)")

        finally:
            # 登出baostock
            bs.logout()
            logger.info("baostock已登出")

    except FileNotFoundError:
        logger.error(f"文件未找到: {csv_file}")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")


if __name__ == "__main__":
    main()