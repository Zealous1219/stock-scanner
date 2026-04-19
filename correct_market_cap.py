"""修正市值单位，将所有市值转换为亿元"""

import pandas as pd
import os

def correct_market_cap_units_and_summary():
    """修正市值单位并生成汇总报告"""

    # 读取包含市值信息的CSV文件
    input_file = "output/black_horse_candidates_with_market_cap_2026-04-15.csv"

    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    # 读取CSV文件
    df = pd.read_csv(input_file)

    print("=" * 100)
    print("股票市值信息汇总 (按市值从大到小排列，单位：亿元)")
    print("=" * 100)

    # 分离成功获取市值和未获取市值的股票
    df_with_market_cap = df[df['market_cap'].notna()].copy()
    df_without_market_cap = df[df['market_cap'].isna()].copy()

    # 将市值从元转换为亿元（除以100000000）
    df_with_market_cap['market_cap_billion'] = df_with_market_cap['market_cap'] / 100000000

    # 按市值从大到小排序
    df_with_market_cap_sorted = df_with_market_cap.sort_values('market_cap_billion', ascending=False)

    print(f"\n成功获取市值的股票 ({len(df_with_market_cap)}只):")
    print("-" * 100)
    print(f"{'排名':<4} {'代码':<12} {'名称':<20} {'市值(亿元)':<15} {'市盈率':<10} {'行业':<30}")
    print("-" * 100)

    for i, (idx, row) in enumerate(df_with_market_cap_sorted.iterrows(), 1):
        code = row['code']
        name = row['name'] if pd.notna(row['name']) else "未知"
        market_cap = f"{row['market_cap_billion']:,.2f}"
        pe_ratio = f"{row['pe_ratio']:.2f}" if pd.notna(row['pe_ratio']) else "N/A"
        industry = row['industry'] if pd.notna(row['industry']) else "未知"

        print(f"{i:<4} {code:<12} {name[:18]:<20} {market_cap:<15} {pe_ratio:<10} {industry[:28]:<30}")

    print(f"\n未获取到市值的股票 ({len(df_without_market_cap)}只):")
    print("-" * 60)
    print(f"{'代码':<12} {'名称':<30} {'原因'}")
    print("-" * 60)

    for idx, row in df_without_market_cap.iterrows():
        code = row['code']
        name = row['name'] if pd.notna(row['name']) else "未知"

        # 判断可能的原因
        reason = "ETF基金或特殊品种" if "ETF" in str(name) or "QDII" in str(name) else "数据缺失"

        print(f"{code:<12} {name[:28]:<30} {reason}")

    # 统计信息
    print(f"\n{'='*100}")
    print("统计信息:")
    print(f"{'='*100}")
    print(f"总股票数量: {len(df)} 只")
    print(f"成功获取市值: {len(df_with_market_cap)} 只 ({len(df_with_market_cap)/len(df)*100:.1f}%)")
    print(f"未获取市值: {len(df_without_market_cap)} 只 ({len(df_without_market_cap)/len(df)*100:.1f}%)")

    # 市值分布统计（单位：亿元）
    if len(df_with_market_cap) > 0:
        print(f"\n市值分布 (单位：亿元):")
        print(f"{'-'*50}")

        # 定义市值区间（亿元）
        ranges = [
            (0, 50, "<50亿"),
            (50, 100, "50-100亿"),
            (100, 200, "100-200亿"),
            (200, 500, "200-500亿"),
            (500, 1000, "500-1000亿"),
            (1000, float('inf'), ">1000亿")
        ]

        distribution_counts = {label: 0 for _, _, label in ranges}

        for idx, row in df_with_market_cap.iterrows():
            market_cap_val = row['market_cap_billion']

            for min_val, max_val, label in ranges:
                if max_val == float('inf'):
                    if market_cap_val >= min_val:
                        distribution_counts[label] += 1
                        break
                else:
                    if min_val <= market_cap_val < max_val:
                        distribution_counts[label] += 1
                        break

        for label in distribution_counts:
            count = distribution_counts[label]
            if count > 0:
                percentage = count / len(df_with_market_cap) * 100
                print(f"{label}: {count} 只 ({percentage:.1f}%)")

    # 市值前十详细分析
    print(f"\n市值前十详细分析 (单位：亿元):")
    print(f"{'-'*100}")
    print(f"{'排名':<4} {'代码':<12} {'名称':<20} {'市值':<12} {'市盈率':<10} {'总股本(亿股)':<12} {'股价':<8}")
    print(f"{'-'*100}")

    top_10 = df_with_market_cap_sorted.head(10)
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        code = row['code']
        name = row['name'] if pd.notna(row['name']) else "未知"
        market_cap = row['market_cap_billion']
        pe_ratio = f"{row['pe_ratio']:.2f}" if pd.notna(row['pe_ratio']) else "N/A"

        # 总股本（万股转换为亿股）
        total_share = f"{row['total_share'] / 10000:,.2f}" if pd.notna(row['total_share']) else "N/A"

        # 股价
        close_price = f"{row['close_price']:.2f}" if pd.notna(row['close_price']) else "N/A"

        print(f"{i:<4} {code:<12} {name[:18]:<20} {market_cap:>11,.2f} {pe_ratio:>10} {total_share:>12} {close_price:>8}")

    # 保存修正后的文件
    output_file = "output/black_horse_candidates_with_market_cap_corrected_2026-04-15.csv"
    df_with_market_cap_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n{'='*100}")
    print(f"修正后的数据已保存到: {output_file}")
    print("注: 市值单位已统一转换为亿元")
    print("总股本单位：亿股（由万股转换而来）")
    print("市盈率基于最新财务报告的净利润计算，负值或缺失表示净利润为负或数据不可用")
    print(f"{'='*100}")

if __name__ == "__main__":
    correct_market_cap_units_and_summary()