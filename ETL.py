#%%
#region[rgba(231,76,60,0.15)]
## Colored Regions

#### 先設置絕對路徑

import os

# 1. 定義你的目標資料夾路徑（前方加上 r 可以避免 Windows 反斜線引發的轉義字元錯誤）
target_path = r"D:\git\Default_risk_ML\data"

# 2. 強制將 Python 的工作目錄切換到該資料夾
os.chdir(target_path)

# 3. 驗證目前的工作目錄（執行時可以順便確認有沒有切換成功）
print("【系統提示】目前的執行路徑已成功設定為：", os.getcwd())

# --- 接下來就可以直接用「相對路徑」讀取 DuckDB 或其他檔案了 ---
# 例如：
# import duckdb
# df = duckdb.query("SELECT * FROM 'your_file.csv' LIMIT 5").df()

#endregion
#%%
## 輸出欄位名稱
import polars as pl
import pandas as pd

df = pl.scan_parquet("train_static_cb_0.parquet")
columns_list = df.columns

columns_list = pd.DataFrame(columns_list)
columns_list.to_csv("QQ.csv", index=False, encoding="utf-8-sig")
columns_list

#%%
#region[rgba(241,196,60,0.15)]

#### 資料展示

import polars as pl
import tkinter as tk
# pip install pandastable
from pandastable import Table


df = pl.scan_parquet("train_static_cb_0.parquet")
#print(df.head(5).collect())
#print(df.head(5).collect().glimpse())

# 1. 建立 Tkinter 視窗
root = tk.Tk()
root.title("資料完整預覽視窗")
root.geometry("1000x600")

# 2. 取出資料
preview_df = df.head(100).collect().to_pandas()

# 3. 渲染表格
frame = tk.Frame(root)
frame.pack(fill='both', expand=True)
pt = Table(frame, dataframe=preview_df, showtoolbar=True)
pt.show()

# 4. 讓視窗持續顯示
root.mainloop()

#endregion
#%%
#region[rgba(46,204,113,0.15)]

### 檢查類別變數
df = pl.scan_parquet("bureau_a2_merge.parquet")
COLUMN = "擔保品估價類型（存續中合約）。"

result = (
    df
    .group_by(COLUMN)
    .agg(pl.len().alias("count"))
    .collect()
)

result = (
    result
    .with_columns(
        (pl.col("count") / pl.col("count").sum() * 100)
        .round(2)
        .alias("percentage")
    )
    .sort("count", descending=True)
)

print(result)

#endregion
#%%
#region[rgba(155,89,182,0.15)]

### 檢查連續變數

df = pl.scan_parquet("bureau_a2_merge.parquet")
COLUMN = "依聯徵中心，已終止合約該筆繳款的逾期天數（num_group1－已終止合約，num_group2－繳款）。"

result = (
    df
    .select([
        pl.len().alias("total_rows"),
        (pl.col(COLUMN) < 0).sum().alias("negative_count")
    ])
    .collect()
)

result = result.with_columns(
    (
    df
    .select([
        pl.col(COLUMN).min().alias("min"),
        pl.col(COLUMN).quantile(0.25).alias("p25"),
        pl.col(COLUMN).quantile(0.5).alias("med"),
        pl.col(COLUMN).quantile(0.75).alias("p75"),
        pl.col(COLUMN).quantile(0.99).alias("p99"),
        pl.col(COLUMN).max().alias("max")

    ])
    .collect()
    )
)

print(result)

#endregion

#%%
#region[rgba(230,126,34,0.15)]

######### 檢查聚合後是否正確

codex = pl.scan_parquet("bureau_a2_aggregated.parquet")
df = pl.scan_parquet("bureau_a2_merge.parquet")

DPD_COL = "存續合約該筆繳款的逾期天數（num_group1－現有合約，num_group2－繳款）。"


case_id_value = 1265705
num_group1_value = 1
## 逾期天數離群值4877 的ID是 1265705 group1 = 1
## 離群值確認未列入計算 ##
## (181554, 0)
## (1504404, 0)
## (1820940, 1)


raw_detail = (
    df
    .filter(
        (pl.col("case_id") == case_id_value) &
        (pl.col("num_group1") == num_group1_value)
    )
    .select([
        "case_id",
        "num_group1",
        "num_group2",
        DPD_COL
    ])
    .sort("num_group2")
    .collect()
)


codex_check = (
    codex
    .filter(
        (pl.col("case_id") == case_id_value) &
        (pl.col("num_group1") == num_group1_value)
    )
    .collect()
)

codex_check

x = raw_detail.to_pandas()

# 1. 建立 Tkinter 視窗
root = tk.Tk()
root.title("資料完整預覽視窗")
root.geometry("1000x600")

# 3. 渲染表格
frame = tk.Frame(root)
frame.pack(fill='both', expand=True)
pt = Table(frame, dataframe=x, showtoolbar=True)
pt.show()

# 4. 讓視窗持續顯示
root.mainloop()

#endregion
# %%
#region[rgba(52,152,219,0.15)]

###### 第0層Merge

import duckdb
con = duckdb.connect()

## bureau_a0 合到 base 生成 base1

# con.sql("""
# COPY(
#         Select *
#         from 'train_base.parquet' as a1
#         left join
#         'bureau_a0_english.parquet' as a2
#         USING(case_id)
#         )to 'base1.parquet' (FORMAT PARQUET)

# """)

## train_static_0 合到 base1 生成 base2

# con.sql("""
# COPY(
#         Select *
#         from 'base1.parquet' as a1
#         left join
#         'train_static_0_high_importance.parquet' as a2
#         USING(case_id)
#         )to 'base2.parquet' (FORMAT PARQUET)

# """)

## train_static_cb_0 合到 base2 生成 base3

# con.sql("""
# COPY(
#         Select *
#         from 'base2.parquet' as a1
#         left join
#         'train_static_cb_0_high_importance.parquet' as a2
#         USING(case_id)
#         )to 'base3.parquet' (FORMAT PARQUET)

# """)

## train_other 合到 base3 生成 base4

# 先處理train_other

con.sql("""
        COPY(
        select *
        from
        (Select case_id , amtdepositbalance_4809441A
         from 'train_other_1.parquet') as t2
        right join 
        'base3.parquet' as t1
        USING(case_id) )to 'base4.parquet' (FORMAT PARQUET)

 """)







con.sql("""
SELECT COUNT(*)
FROM (
    DESCRIBE
    SELECT *
    FROM 'base4.parquet'
);
""")


#endregion
# %%
