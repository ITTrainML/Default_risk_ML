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

df = pl.scan_parquet("base9.parquet")
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


df = pl.scan_parquet("base8.parquet")
#print(df.head(5).collect())
#print(df.head(5).collect().glimpse())

# 1. 建立 Tkinter 視窗
root = tk.Tk()
root.title("資料完整預覽視窗")
root.geometry("1000x600")

# 2. 取出資料
preview_df = df.head(1000).collect().to_pandas()

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

# 順便處理train_other，僅保留一個欄位後再合併

# con.sql("""
#         COPY(
#         select *
#         from
#         (Select case_id , amtdepositbalance_4809441A
#          from 'train_other_1.parquet') as t2
#         right join 
#         'base3.parquet' as t1
#         USING(case_id) )to 'base4.parquet' (FORMAT PARQUET)

#  """)

## train_debitcard_agg 合到 base4 生成 base5

# con.sql("""
# COPY(
#         Select *
#         from 'base4.parquet' as a1
#         left join
#         'train_debitcard_agg.parquet' as a2
#         USING(case_id)
#         )to 'base5.parquet' (FORMAT PARQUET)

# """)

## train_tax_registry_a_agg 合到 base5 生成 base6

# con.sql("""
# COPY(
#         Select *
#         from 'base5.parquet' as a1
#         left join
#         'train_tax_registry_a_agg.parquet' as a2
#         USING(case_id)
#         )to 'base6.parquet' (FORMAT PARQUET)

# """)

## train_tax_registry_b_agg 合到 base6 生成 base7

# con.sql("""
# COPY(
#         Select *
#         from 'base6.parquet' as a1
#         left join
#         'train_tax_registry_b_agg.parquet' as a2
#         USING(case_id)
#         )to 'base7.parquet' (FORMAT PARQUET)

# """)

## train_tax_registry_c_agg 合到 base7 生成 base8

# con.sql("""
# COPY(
#         Select *
#         from 'base7.parquet' as a1
#         left join
#         'train_tax_registry_c_agg.parquet' as a2
#         USING(case_id)
#         )to 'base8.parquet' (FORMAT PARQUET)

# """)

## train_applprev_agg 合到 base8 生成 base9

con.sql("""
COPY(
        Select *
        from 'base8.parquet' as a1
        left join
        'train_applprev_agg.parquet' as a2
        USING(case_id)
        )to 'base9.parquet' (FORMAT PARQUET)

""")




con.sql("""
SELECT COUNT(*)
FROM (
    DESCRIBE
    SELECT *
    FROM 'base9.parquet'
);
""")


#endregion
# %%
#region[rgba(231,76,60,0.15)]

## 處理深度1的表

# 定義一個封裝 Polars 表達式的函數

import polars as pl


def aggregate_numeric_features(
    lf: pl.LazyFrame,
    group_keys: str,
    cols: list[str],
    group1_col: str | None = "num_group1"
) -> pl.LazyFrame:
    """
    通用數值聚合

    Parameters
    ----------
    lf : LazyFrame
    group_keys : 分群key，例如
        ["case_id"]
        ["case_id","num_group1"]
    cols : 要聚合的欄位

    Returns
    -------
    LazyFrame
    """

    exprs: list[pl.Expr] = []

    # num_group1 僅計算紀錄／合約數量
    if group1_col is not None:
        exprs.append(
            pl.col(group1_col)
            .n_unique()
            .alias("debitcard_1_count")
        )

    for c in cols:

        exprs.extend([

            # ---------------------
            # 基本統計
            # ---------------------

            pl.col(c).mean().alias(f"{c}_mean"),

            pl.col(c).max().alias(f"{c}_max"),

            pl.col(c).median().alias(f"{c}_median"),

            # ---------------------
            # Positive
            # ---------------------

            (pl.col(c) > 0)
                .sum()
                .alias(f"{c}_positive_count"),

            (
                (pl.col(c) > 0)
                .sum()
                /
                pl.col(c).is_not_null().sum()
            ).alias(f"{c}_overdue_rate"),

            (
                pl.when(pl.col(c) > 0)
                .then(pl.col(c))
                .otherwise(0)
                .sum()
            ).alias(f"{c}_sum_positive"),

            (
                pl.when(pl.col(c) > 0)
                .then(pl.col(c))
                .otherwise(None)
                .mean()
            ).alias(f"{c}_mean_positive"),

            # ---------------------
            # Stability
            # ---------------------

            pl.col(c).std().alias(f"{c}_std"),

            (
                pl.col(c).quantile(0.75)
                -
                pl.col(c).quantile(0.25)
            ).alias(f"{c}_iqr"),

            # ---------------------
            # Missing
            # ---------------------

            pl.col(c)
                .is_null()
                .sum()
                .alias(f"{c}_null_count"),

            pl.col(c)
                .is_null()
                .mean()
                .alias(f"{c}_null_rate"),

            pl.col(c)
                .is_not_null()
                .sum()
                .alias(f"{c}_non_null_count"),
          
        ])

    return (
        lf
        .group_by(group_keys)
        .agg(exprs)
    )

### 讀取檔案並刪除指定欄位後 套用函式

lf = pl.scan_parquet("train_debitcard_1.parquet")
 # 刪除欄位
lf = lf.drop([
    "openingdate_857D",

])


data_cols = [
    "last180dayaveragebalance_704A",
    "last180dayturnover_1134A",
    "last30dayturnover_651A"
]

agg = aggregate_numeric_features(
    lf,
    group_keys=["case_id"],
    cols=data_cols,
    group1_col="num_group1"
)

agg.sink_parquet("train_debitcard_agg.parquet")

#endregion
# %%
#region[rgba(46,204,113,0.15)]

import polars as pl

base = pl.scan_parquet("base9.parquet")

## 先確認Target 分布比例
# target_summary = (
#     base
#     .group_by("target")
#     .len()
#     .sort("target")
#     .collect()
# )

# print(target_summary)


## 開始抽樣

N_DEFAULT = 40_000
N_NON_DEFAULT = 40_000
N_ROUNDS = 30
BASE_SEED = 20260717

sampling_base = (
    base
    .select(["case_id", "target"])
    .collect()
)

#建立違約與非違約 ID 母體
default_ids = (
    sampling_base
    .filter(pl.col("target") == 1)
    .select("case_id")
)

non_default_ids = (
    sampling_base
    .filter(pl.col("target") == 0)
    .select("case_id")
)


#把所有非違約者打亂

non_default_ids_shuffled = non_default_ids.sample(
    fraction=1.0,
    with_replacement=False,
    shuffle=True,
    seed=BASE_SEED,
)

#函數先產生本輪的 80,000 個 case_id，
#再回到原始 LazyFrame 取完整資料。

def get_sample_round_lazy(
    round_idx: int,
    base_lf: pl.LazyFrame,
    default_ids: pl.DataFrame,
    non_default_ids_shuffled: pl.DataFrame,
    n_default: int = 40_000,
    n_non_default: int = 40_000,
    base_seed: int = 20260717,
) -> pl.DataFrame:
    """
    從 LazyFrame base 中建立單輪完整樣本。

    round_idx 從 0 開始。

    違約者：
    - 每輪內不重複
    - 不同輪之間可以重複

    非違約者：
    - 30輪之間完全不重複
    """

    if round_idx < 0:
        raise ValueError("round_idx 不得小於 0。")

    if default_ids.height < n_default:
        raise ValueError("違約樣本數不足。")

    non_default_start = round_idx * n_non_default
    non_default_end = non_default_start + n_non_default

    if non_default_end > non_default_ids_shuffled.height:
        raise ValueError(
            f"第 {round_idx + 1} 輪超出非違約母體範圍。"
        )

    # 每輪重新抽違約者
    default_sample_ids = default_ids.sample(
        n=n_default,
        with_replacement=False,
        shuffle=True,
        seed=base_seed + round_idx,
    )

    # 非違約者從已打亂的母體切不同區段
    non_default_sample_ids = non_default_ids_shuffled.slice(
        offset=non_default_start,
        length=n_non_default,
    )

    # 合併本輪80,000個case_id
    round_ids = pl.concat(
        [
            default_sample_ids,
            non_default_sample_ids,
        ],
        how="vertical",
    )

    # 加上順序欄位，方便最後重新隨機排列
    round_ids = (
        round_ids
        .sample(
            fraction=1.0,
            with_replacement=False,
            shuffle=True,
            seed=base_seed + 10_000 + round_idx,
        )
        .with_row_index("sample_order")
    )

    # 用case_id回到原始LazyFrame抓完整特徵
    sample_df = (
        base_lf
        .join(
            round_ids.lazy(),
            on="case_id",
            how="inner",
        )
        .sort("sample_order")
        .drop("sample_order")
        .collect()
    )

    return sample_df


sample_01 = get_sample_round_lazy(
    round_idx=0,
    base_lf=base,
    default_ids=default_ids,
    non_default_ids_shuffled=non_default_ids_shuffled,
    n_default=N_DEFAULT,
    n_non_default=N_NON_DEFAULT,
    base_seed=BASE_SEED,
)


sample_01.group_by("target").len().sort("target")








### 跑30次迴圈

# for round_idx in range(N_ROUNDS):

#     sample_df = get_sample_round_lazy(
#         round_idx=round_idx,
#         base_lf=base,
#         default_ids=default_ids,
#         non_default_ids_shuffled=non_default_ids_shuffled,
#         n_default=N_DEFAULT,
#         n_non_default=N_NON_DEFAULT,
#         base_seed=BASE_SEED,
#     )

#     validate_sample(
#         sample_df,
#         n_default=N_DEFAULT,
#         n_non_default=N_NON_DEFAULT,
#     )

#     print(
#         f"開始第 {round_idx + 1:02d} 輪特徵篩選"
#     )

#     # 此時sample_df才是完整80,000人 × 約1,000欄的DataFrame
#     X = sample_df.drop(["case_id", "target"])
#     y = sample_df["target"]

#     # 在此跑模型


#endregion
# %%
