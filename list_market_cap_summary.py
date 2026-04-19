"""列出所有股票的市值信息，按市值从大到小排列"""

import pandas as pd
import os

def list_market_cap_summary():
    """列出所有股票的市值信息"""

    # 读取包含市值信息的CSV文件
    input_file = "output/black_horse_candidates_with_market_cap_2026-04-15.csv"

    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    # 读取CSV文件
    df = pd.read_csv(input_file)

    print("=" * 80)
    print("股票市值信息汇总 (按市值从大到小排列)")
    print("=" * 80)

    # 分离成功获取市值和未获取市值的股票
    df_with_market_cap = df[df['market_cap'].notna()].copy()
    df_without_market_cap = df[df['market_cap'].isna()].copy()

    # 按市值从大到小排序
    df_with_market_cap_sorted = df_with_market_cap.sort_values('market_cap', ascending=False)

    print(f"\n成功获取市值的股票 ({len(df_with_market_cap)}只):")
    print("-" * 80)
    print(f"{'排名':<4} {'代码':<12} {'名称':<20} {'市值(亿元)':<15} {'市盈率':<10} {'行业':<30}")
    print("-" * 80)

    for i, (idx, row) in enumerate(df_with_market_cap_sorted.iterrows(), 1):
        code = row['code']
        name = row['name'] if pd.notna(row['name']) else "未知"
        market_cap = f"{row['market_cap']:,.2f}" if pd.notna(row['market_cap']) else "N/A"
        pe_ratio = f"{row['pe_ratio']:.2f}" if pd.notna(row['pe_ratio']) else "N/A"
        industry = row['industry'] if pd.notna(row['industry']) else "未知"

        print(f"{i:<4} {code:<12} {name[:18]:<20} {market_cap:<15} {pe_ratio:<10} {industry[:28]:<30}")

    print(f"\n未获取到市值的股票 ({len(df_without_market_cap)}只):")
    print("-" * 80)
    print(f"{'代码':<12} {'名称':<30} {'原因'}")
    print("-" * 80)

    for idx, row in df_without_market_cap.iterrows():
        code = row['code']
        name = row['name'] if pd.notna(row['name']) else "未知"

        # 判断可能的原因
        reason = "ETF基金或特殊品种" if "ETF" in str(name) or "QDII" in str(name) else "数据缺失"

        print(f"{code:<12} {name[:28]:<30} {reason}")

    # 统计信息
    print(f"\n{'='*80}")
    print("统计信息:")
    print(f"{'='*80}")
    print(f"总股票数量: {len(df)} 只")
    print(f"成功获取市值: {len(df_with_market_cap)} 只 ({len(df_with_market_cap)/len(df)*100:.1f}%)")
    print(f"未获取市值: {len(df_without_market_cap)} 只 ({len(df_without_market_cap)/len(df)*100:.1f}%)")

    # 市值分布统计
    if len(df_with_market_cap) > 0:
        print(f"\n市值分布:")
        print(f"{'-'*40}")

        # 定义市值区间
        ranges = [
            (0, 50, "<50亿"),
            (50, 100, "50-100亿"),
            (100, 200, "100-200亿"),
            (200, 500, "200-500亿"),
            (500, 1000, "500-1000亿"),
            (1000, float('inf'), ">1000亿")
        ]

        for min_val, max_val, label in ranges:
            if max_val == float('inf'):
                count = len(df_with_market_cap[df_with_market_cap['market_cap'] >= min_val])
            else:
                count = len(df_with_market_cap[(df_with_market_cap['market_cap'] >= min_val) &
                                               (df_with_market_cap['market_cap'] < max_val)])

            if count > 0:
                percentage = count / len(df_with_market_cap) * 100
                print(f"{label}: {count} 只 ({percentage:.1f}%)")

    # 行业分布
    if len(df_with_market_cap) > 0:
        industry_counts = df_with_market_cap['industry'].value_counts().head(10)

        print(f"\n行业分布 (前10):")
        print(f"{'-'*40}")

        for industry, count in industry_counts.items():
            if pd.notna(industry):
                percentage = count / len(df_with_market_cap) * 100
                print(f"{industry[:35]:<35} {count:>2} 只 ({percentage:.1f}%)")

    print(f"\n{'='*80}")
    print("注: 市值基于最新财务报告的总股本和最新收盘价计算")
    print("市盈率基于最新财务报告的净利润计算，负值或缺失表示净利润为负或数据不可用")
    print(f"{'='*80}")

if __name__ == "__main__":
    list_market_cap_summary()