#%%
#region[rgba(231,76,60,0.15)]

#### 先設置絕對路徑

import os

# 1. 定義你的目標資料夾路徑（前方加上 r 可以避免 Windows 反斜線引發的轉義字元錯誤）
target_path = r"C:\Users\ittraining\Desktop\git\Default_risk_ML\data"

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
#region[rgba(241,196,60,0.15)]

#### 資料展示

import polars as pl
import tkinter as tk
from pandastable import Table

df = pl.scan_parquet("bureau_a2_merge.parquet")
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
# %%
#region[rgba(52,152,219,0.15)]



# 1. 關鍵設定：將最大顯示欄位與列數設為 None (代表不限制)
pl.Config.set_tbl_cols(-1)  # 顯示所有欄位（或設為 None / -1）
pl.Config.set_tbl_rows(-1)  # 顯示所有列（或設為 None / -1）

# 2. 如果你的欄位非常多、或是欄位內的文字很長，也可以加上這兩個：
pl.Config.set_fmt_str_lengths(100) # 每個字串欄位最多顯示幾個字（預設通常會被切斷）
pl.Config.set_tbl_width_chars(2000) # 終端機整行的字元寬度限制（設大一點避免換行錯位）


### 1. 檢查聚合後 key 是否唯一

# is_unique = (
#     df
#     .group_by(["case_id", "num_group1"])
#     .agg(pl.len().alias("count"))
#     .select(
#         (
#             pl.col("count") == 1
#         ).all().alias("is_unique")
#     )
#     .collect()
# )

# print(is_unique)

### 2. 檢查聚合後筆數是否等於 raw 的 group 數

# bureau_a2_raw = pl.scan_parquet("bureau_a2_merge.parquet")

# check_rows = pl.DataFrame({
#     "source": ["raw_group_count", "agg_row_count"],
#     "rows": [
#         bureau_a2_raw.select(
#             pl.struct(["case_id", "num_group1"]).n_unique()
#         ).collect().item(),
#         df.select(pl.len()).collect().item()
#     ]
# })

# print(check_rows)


### 3. 隨機抽幾組 key，手動重算對照

sample_keys = (
    df
    .filter(
        pl.col("active_dpd_mean_all") != pl.col("active_dpd_mean_positive")
    )
    .select(["case_id", "num_group1"])
    .limit(100000)
    .collect()
    .sample(n=10, seed=50)
)

print(sample_keys)

case_id_value = sample_keys["case_id"][1]
num_group1_value = sample_keys["num_group1"][1]


DPD_COL = "存續合約該筆繳款的逾期天數（num_group1－現有合約，num_group2－繳款）。"

raw_detail = (
    bureau_a2_raw
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
    df
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
