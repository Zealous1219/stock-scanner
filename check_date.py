import baostock as bs
import pandas as pd

lg = bs.login()
rs = bs.query_hs300_stocks()
data_list = []
while rs.error_code == '0' and rs.next():
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)
print(f"沪深300成分股更新日期: {df['updateDate'].iloc[0]}")

rs2 = bs.query_history_k_data_plus(
    "sh.600000",
    "date,code,open,high,low,close,volume",
    start_date='2026-01-01',
    end_date='2026-12-31',
    frequency="d"
)
data_list2 = []
while rs2.error_code == '0' and rs2.next():
    data_list2.append(rs2.get_row_data())
if data_list2:
    df2 = pd.DataFrame(data_list2, columns=rs2.fields)
    print(f"上证指数最新数据日期: {df2['date'].max()}")
bs.logout()
