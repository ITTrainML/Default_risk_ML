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


df = pl.scan_parquet("train_base_final3.parquet")
#print(df.head(5).collect())
#print(df.head(5).collect().glimpse())

df1 = df.collect()
print(df1.shape)

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

# con.sql("""
# COPY(
#         Select *
#         from 'base8.parquet' as a1
#         left join
#         'train_applprev_agg.parquet' as a2
#         USING(case_id)
#         )to 'base9.parquet' (FORMAT PARQUET)

# """)

## train_credit_bureau_b1b2_case_aggregated 合到 base9 生成 base10

con.sql("""
COPY(
        Select *
        from 'base9.parquet' as a1
        left join
        'train_credit_bureau_b1b2_case_aggregated.parquet' as a2
        USING(case_id)
        )to 'base10.parquet' (FORMAT PARQUET)

""")





con.sql("""
SELECT COUNT(*)
FROM (
    DESCRIBE
    SELECT *
    FROM 'base10.parquet'
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
#region[rgba(241,196,60,0.15)]

import polars as pl

lf = pl.scan_parquet("base_final.parquet")

selected_features = [
"pmts_dpd_303P_std__mean",
"avgdpdtolclosure24_3658938P",
"maxdbddpdtollast12m_3658940P",
"pctinstlsallpaidlate1d_3546856L",
"reject_rate",
"dpdmax_139P__mean",
"dpdmax_757P__mean",
"dpdmax_757P__std",
"numberofoverdueinstlmax_1039L__mean",
"numberofoverdueinstlmax_1151L__mean",
"overdueamountmax_155A__mean",
"pmts_dpd_1073P_std__mean",
"pmts_overdue_1140A_mean__mean",
"pmts_dpd_1073P_trend__max",
"pmts_dpd_1073P_longest_good_streak__max",
"pmts_dpd_303P_longest_good_streak__max",
"pctinstlsallpaidearl3d_427L",
"maxdpdtolerance_577P_mean",
"rejectreasonclient_4145042M_non_placeholder_ratio",
"pmts_overdue_1140A_std__mean",
"lastrejectreason_759M",
"dpdmax_139P__sum",
"numberofoverdueinstlmax_1039L__max",
"numberofoverdueinstlmax_1039L__sum",
"numberofoverdueinstlmax_1151L__std",
"pmts_dpd_1073P_overdue_rate__weighted_avg",
"pmts_dpd_1073P_non_null_count__max",
"pmts_dpd_303P_mean__mean",
"pmts_dpd_303P_overdue_rate__weighted_avg",
"pmts_overdue_1140A_overdue_rate__weighted_avg",
"pmts_overdue_1140A_sum_positive__sum",
"pmts_overdue_1152A_overdue_rate__weighted_avg",
"pmts_dpd_1073P_trend__mean",
"pmts_dpd_303P_trend__mean",
"avgdbddpdlast24m_3658932P",
"cntpmts24_3658933L",
"credamount_770A",
"disbursedcredamount_1113A",
"maxdpdlast12m_727P",
"maxdpdlast24m_143P",
"maxdpdlast9m_1059P",
"monthsannuity_845L",
"numrejects9m_859L",
"price_1097A",
"education_1103M",
"pmtaverage_3A",
"pmtssum_45A",
"requesttype_4525192L",
"tenure_years_max",
"age_years_appl",
"tenure_years_appl",
"education_927M_appl",
"incometype_1044T_appl",
"registaddr_district_1083M_appl",
"date_decision",
"dpdmax_757P__max",
"dpdmax_757P__sum",
"numberofoutstandinstls_59L__min",
"numberofoutstandinstls_59L__sum",
"numberofoverdueinstlmax_1151L__max",
"numberofoverdueinstlmax_1151L__sum",
"overdueamountmax2_14A__mean",
"overdueamountmax2_14A__max",
"overdueamountmax_155A__max",
"overdueamountmax_155A__sum",
"overdueamountmax_35A__mean",
"overdueamountmax_35A__std",
"pmts_dpd_1073P_std__max",
"pmts_dpd_303P_mean_positive__recomputed",
"pmts_overdue_1140A_std__max",
"pmts_dpd_1073P_consecutive_max__max",
"pmts_dpd_303P_recent3_mean__mean",
"pmts_dpd_303P_recent6_mean__mean",
"pmts_dpd_303P_recent6_mean__max",
"pmts_dpd_303P_recent12_mean__mean",
"interestrate_311L",
"maxdbddpdlast1m_3658939P",
"maxdebt4_972A",
"maxdpdtolerance_374P",
"numinstlswithdpd10_728L",
"numinstlswithdpd5_4187116L",
"pctinstlsallpaidlate4d_3546849L",
"totaldebt_9A",
"totalsettled_863A",
"n_rejected",
"numberofoverdueinstlmax_1039L__min",
"prolongationcount_599L__null_rate",
"residualamount_856A__mean",
"residualamount_856A__max",
"residualamount_856A__min",
"residualamount_856A__sum",
"totalamount_6A__sum",
"totalamount_996A__mean",
"pmts_dpd_303P_non_null_count__sum",
"pmts_overdue_1140A_sum_positive__max",
"collater_valueofguarantee_1124L_null_count__max",
"pmts_dpd_303P_recent_time_key__mean_fallback",
"currdebt_22A",
"disbursementtype_67L",
"eir_270L",
"maxdbddpdtollast6m_4187119P",
"maxdpdlast3m_392P",
"maxdpdlast6m_474P",
"numinstlswithoutdpd_562L",
"numinsttopaygr_769L",
"numinstunpaidmax_3546851L",
"numinstunpaidmaxest_4493212L",
"pctinstlsallpaidlat10d_839L",
"days120_123L",
"days180_256L",
"days30_165L",
"days360_512L",
"days90_310L",
"tax_registry_a_count",
"amount_4527230A_sum_positive",
"tax_registry_c_count",
"pmtamount_36A_positive_count",
"pmtamount_36A_sum_positive",
"cancelreason_3545846M_mode",
"education_1138M_mode",
"rejectreason_755M_non_placeholder_ratio",
"contaddr_district_15M_appl",
"familystate_447L_appl",
"relationshiptoclient_415T_mode",
"relationshiptoclient_642T_mode",
"dpdmax_139P__max",
"dpdmax_757P__null_rate",
"monthlyinstlamount_674A__mean",
"outstandingamount_362A__max",
"overdueamountmax2_14A__sum",
"overdueamountmax2_398A__mean",
"overdueamountmax2_398A__max",
"overdueamountmax2_398A__std",
"overdueamountmax_155A__std",
"prolongationcount_599L__sum",
"totalamount_6A__null_rate",
"totalamount_996A__max",
"totaloutstanddebtvalue_39A__sum",
"pmts_dpd_1073P_mean__mean",
"pmts_dpd_1073P_mean__max",
"pmts_dpd_1073P_max__max",
"pmts_dpd_1073P_n_unique__max",
"pmts_dpd_1073P_positive_count__sum",
"pmts_dpd_1073P_positive_count__max",
"pmts_dpd_1073P_mean_positive__max",
"pmts_dpd_1073P_sum_positive__sum",
"pmts_dpd_1073P_sum_positive__max",
"pmts_dpd_1073P_non_null_count__sum",
"pmts_dpd_303P_mean__max",
"pmts_dpd_303P_max__max",
"maxannuity_4075009A",
"pmtamount_36A_non_null_count",
"empl_industry_691L_appl",
"amount_4527230A_positive_count",
"numberofqueries_373L",
"credtype_322L",
"last_status",
"riskassesment_940T",
"b2_pmts_dpdvalue_108P_overdue_rate_recomputed",
"b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed",
"b2_pmts_dpdvalue_108P_overdue_rate_contract_max",
"b2_pmts_pmtsoverdue_635A_mean_weighted",
"b2_pmts_pmtsoverdue_635A_mean_contract_mean",
"b2_pmts_pmtsoverdue_635A_overdue_rate_contract_max",
"b2_pmts_dpdvalue_108P_mean_weighted",
"b2_pmts_pmtsoverdue_635A_mean_contract_max",
"b2_pmts_dpdvalue_108P_mean_contract_mean",
"pmts_dpd_303P_n_unique__sum",
"pmts_overdue_1140A_non_null_count__sum",
"pmts_year_1139T_pmts_month_158T_duration__mean",
"pmts_overdue_1140A_n_unique__sum",
"pmts_dpd_1073P_recent_time_key__mean_fallback",
"pmts_dpd_303P_recent_time_key__max_fallback",
"opencred_647L",
"subjectroles_name_838M_n_unique__sum",
"subjectroles_name_838M_entropy__mean",
"pmts_overdue_1140A_non_null_count__max",
"pmts_overdue_1152A_mean__mean",
"pmts_overdue_1152A_std__mean",
"pmts_dpd_303P_std__max",
"pmts_overdue_1140A_mean__max",
"pmts_dpd_1073P_recent12_mean__mean",
"pmts_dpd_303P_recent12_mean__max",
"pmts_dpd_1073P_recent12_mean__max",
"pmts_overdue_1152A_median__mean",
"pctinstlsallpaidlate6d_3546844L",
"pmts_dpd_303P_median__mean",
"registaddr_zipcode_184M_appl",
"lastrejectdate_50D",
"maxdpdinstldate_3546855D",
"lastdelinqdate_224D",
"datelastunpaid_3546854D",
"pmts_year_507T_pmts_month_706T_min__min",
"pmts_year_1139T_pmts_month_158T_min__min",
"prolongationcount_599L__mean",
]

keep_columns = ["case_id", *selected_features, "target"]

lf_selected = lf.select(keep_columns)

lf_selected.sink_parquet("base_final2.parquet")

#endregion
# %%
#region[rgba(52,152,219,0.15)]

### 做相關係數


from itertools import combinations

import polars as pl


def pairwise_correlations_lazy(
    lf: pl.LazyFrame,
    feature_columns: list[str],
    method: str = "pearson",
    batch_size: int = 500,
) -> pl.DataFrame:
    """
    使用Polars LazyFrame計算所有不重複的兩兩相關係數。

    Parameters
    ----------
    lf:
        Polars LazyFrame。
    feature_columns:
        要計算相關係數的數值特徵名稱。
    method:
        "pearson" 或 "spearman"。
    batch_size:
        每批計算的特徵組合數，避免一次建立過大的查詢。

    Returns
    -------
    pl.DataFrame
        欄位包含：
        - feature_1
        - feature_2
        - correlation
        - abs_correlation
    """

    if method not in {"pearson", "spearman"}:
        raise ValueError("method必須是 'pearson' 或 'spearman'")

    # 去除重複欄位，但維持原順序
    feature_columns = list(dict.fromkeys(feature_columns))

    # 檢查指定欄位是否存在
    available_columns = set(lf.collect_schema().names())

    missing_columns = [
        column
        for column in feature_columns
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"以下 {len(missing_columns)} 個欄位不存在：\n"
            + "\n".join(missing_columns)
        )

    # 產生不重複的兩兩組合
    feature_pairs = list(combinations(feature_columns, 2))

    print(f"特徵數：{len(feature_columns):,}")
    print(f"兩兩組合數：{len(feature_pairs):,}")

    result_frames = []

    # 分批執行
    for start in range(0, len(feature_pairs), batch_size):
        batch_pairs = feature_pairs[start:start + batch_size]

        correlation_expressions = [
            pl.corr(
                pl.col(feature_1),
                pl.col(feature_2),
                method=method,
            ).alias(f"corr_{index}")
            for index, (feature_1, feature_2) in enumerate(batch_pairs)
        ]

        # 此處仍是LazyFrame；直到collect才真正執行
        batch_result = (
            lf.select(feature_columns)
            .select(correlation_expressions)
            .collect()
        )

        # select聚合後只有一列，每欄代表一組相關係數
        correlation_values = batch_result.row(0)

        batch_long = pl.DataFrame(
            {
                "feature_1": [
                    pair[0] for pair in batch_pairs
                ],
                "feature_2": [
                    pair[1] for pair in batch_pairs
                ],
                "correlation": correlation_values,
            }
        ).with_columns(
            pl.col("correlation")
            .abs()
            .alias("abs_correlation")
        )

        result_frames.append(batch_long)

        print(
            f"已完成："
            f"{min(start + batch_size, len(feature_pairs)):,}"
            f" / {len(feature_pairs):,}"
        )

    return pl.concat(result_frames)



lf = pl.scan_parquet("base_final2.parquet")

corr_pairs = pairwise_correlations_lazy(
    lf=lf,
    feature_columns=selected_features,
    method="pearson",
    batch_size=500,
)

corr_pairs.write_csv(
    "correlation_pairs.csv"
)
#endregion
# %%
#region[rgba(155,89,182,0.15)]

### 刪除高相關係數後，做共線性 ### 缺失值要補才能做

# import polars as pl

# lf = pl.scan_parquet("base_final2.parquet")

# lf1 = lf.drop(["pmts_dpd_1073P_std__max",
# "pmts_dpd_1073P_mean__mean",
# "pmts_dpd_1073P_max__max",
# "pmts_dpd_1073P_recent12_mean__mean",
# "pmts_dpd_1073P_recent12_mean__max",
# "pmts_dpd_1073P_sum_positive__max",
# "pmts_dpd_1073P_n_unique__max",
# "pmts_dpd_1073P_positive_count__sum",
# "pmts_dpd_1073P_positive_count__max",
# "numberofoverdueinstlmax_1151L__std",
# "pmts_dpd_303P_mean__max",
# "pmts_dpd_303P_recent12_mean__max",
# "dpdmax_757P__max",
# "numberofoverdueinstlmax_1151L__max",
# "pmts_dpd_303P_recent6_mean__max",
# "pmts_dpd_303P_max__max",
# "pmts_dpd_303P_recent12_mean__mean",
# "numberofoverdueinstlmax_1151L__mean",
# "dpdmax_757P__mean",
# "pmts_dpd_303P_recent6_mean__mean",
# "pmts_dpd_303P_recent3_mean__mean",
# "pmts_dpd_303P_median__mean",
# "dpdmax_139P__max",
# "numberofoverdueinstlmax_1039L__mean",
# "numberofoverdueinstlmax_1039L__sum",
# "tax_registry_a_count",
# "pmtamount_36A_positive_count",
# "tax_registry_c_count",
# "pmtamount_36A_non_null_count",
# "overdueamountmax_155A__max",
# "overdueamountmax2_14A__sum",
# "totalamount_996A__max",
# "totaloutstanddebtvalue_39A__sum",
# "residualamount_856A__mean",
# "residualamount_856A__sum",
# "pmtamount_36A_sum_positive",
# "pmtssum_45A",
# "b2_pmts_pmtsoverdue_635A_mean_contract_mean",
# "days120_123L",
# "maxdbddpdtollast6m_4187119P",
# "maxdbddpdlast1m_3658939P",
# "numinstunpaidmax_3546851L",
# "numinsttopaygr_769L",
# "overdueamountmax2_398A__std",
# "overdueamountmax2_398A__max",
# "pctinstlsallpaidlate6d_3546844L",
# "pctinstlsallpaidlat10d_839L",
# "pmts_overdue_1140A_sum_positive__max",
# "avgdbddpdlast24m_3658932P",
# "b2_pmts_dpdvalue_108P_mean_contract_mean",
# "b2_pmts_pmtsoverdue_635A_overdue_rate_contract_max",
# "b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed",
# "credamount_770A",
# "currdebt_22A",
# "numberofqueries_373L",
# "totalamount_6A__null_rate",
# "numberofoverdueinstlmax_1151L__sum",
# "eir_270L",
# "maxdpdlast9m_1059P",
# "monthsannuity_845L",
# "overdueamountmax2_14A__mean",
# "overdueamountmax2_398A__mean",
# "pmts_dpd_1073P_non_null_count__max",
# "pmts_overdue_1140A_non_null_count__sum",
# "pmts_dpd_1073P_overdue_rate__weighted_avg",
# "pmts_overdue_1152A_overdue_rate__weighted_avg",
# "pmts_overdue_1152A_median__mean",
# "rejectreason_755M_non_placeholder_ratio",
# "case_id",
# "target",
# ])




# import polars.selectors as cs

# # 選擇所有數值欄位
# lf2 = lf1.select(cs.numeric())


# # 抽樣10%

# vif_features = lf2.select(cs.all())

# lf_sample = (
#     lf2
#     # 暫時建立資料列索引
#     .with_row_index("_row_id")

#     # 將列索引雜湊後分成10組，只取其中1組
#     .filter(
#         (
#             pl.col("_row_id").hash(seed=42) % 10
#         ) == 0
#     )

#     # 移除暫時建立的索引
#     .drop("_row_id")
# )

# df_sample_pl = lf_sample.collect()

# df_sample_pd = df_sample_pl.to_pandas()

# print(type(df_sample_pd))
# print(df_sample_pd.shape)


### 做共線性 (未填補缺失值，所以這邊不能做，但可以用WOE去做)

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd


def calculate_vif(df):
    return pd.DataFrame({
        "Feature": df.columns,
        "VIF": [
            variance_inflation_factor(
                df.values,
                i
            )
            for i in range(df.shape[1])
        ]
    }).sort_values(
        "VIF",
        ascending=False
    )

vif = calculate_vif(df_sample_pd)

#endregion
# %%

import polars as pl

lf = pl.scan_parquet("base_final2.parquet")
lf1 = lf.drop(["pmts_dpd_303P_std__mean",
"dpdmax_139P__mean",
"dpdmax_757P__mean",
"dpdmax_757P__std",
"numberofoverdueinstlmax_1039L__mean",
"numberofoverdueinstlmax_1151L__mean",
"overdueamountmax_155A__mean",
"pmts_dpd_1073P_std__mean",
"pmts_overdue_1140A_mean__mean",
"pmts_dpd_1073P_longest_good_streak__max",
"pmts_overdue_1140A_std__mean",
"dpdmax_139P__sum",
"numberofoverdueinstlmax_1039L__max",
"numberofoverdueinstlmax_1039L__sum",
"numberofoverdueinstlmax_1151L__std",
"pmts_dpd_1073P_overdue_rate__weighted_avg",
"pmts_dpd_1073P_non_null_count__max",
"pmts_dpd_303P_mean__mean",
"pmts_dpd_303P_overdue_rate__weighted_avg",
"pmts_overdue_1140A_sum_positive__sum",
"pmts_overdue_1152A_overdue_rate__weighted_avg",
"avgdbddpdlast24m_3658932P",
"credamount_770A",
"maxdpdlast9m_1059P",
"monthsannuity_845L",
"pmtssum_45A",
"registaddr_district_1083M_appl",
"date_decision",
"dpdmax_757P__max",
"dpdmax_757P__sum",
"numberofoverdueinstlmax_1151L__max",
"numberofoverdueinstlmax_1151L__sum",
"overdueamountmax2_14A__mean",
"overdueamountmax2_14A__max",
"overdueamountmax_155A__max",
"overdueamountmax_155A__sum",
"pmts_dpd_1073P_std__max",
"pmts_overdue_1140A_std__max",
"pmts_dpd_1073P_consecutive_max__max",
"pmts_dpd_303P_recent3_mean__mean",
"pmts_dpd_303P_recent6_mean__mean",
"pmts_dpd_303P_recent6_mean__max",
"pmts_dpd_303P_recent12_mean__mean",
"maxdbddpdlast1m_3658939P",
"pctinstlsallpaidlate4d_3546849L",
"totaldebt_9A",
"n_rejected",
"residualamount_856A__mean",
"residualamount_856A__max",
"residualamount_856A__sum",
"pmts_overdue_1140A_sum_positive__max",
"currdebt_22A",
"eir_270L",
"maxdbddpdtollast6m_4187119P",
"maxdpdlast6m_474P",
"numinsttopaygr_769L",
"numinstunpaidmax_3546851L",
"pctinstlsallpaidlat10d_839L",
"days120_123L",
"tax_registry_a_count",
"tax_registry_c_count",
"pmtamount_36A_positive_count",
"pmtamount_36A_sum_positive",
"rejectreason_755M_non_placeholder_ratio",
"contaddr_district_15M_appl",
"relationshiptoclient_415T_mode",
"relationshiptoclient_642T_mode",
"dpdmax_139P__max",
"overdueamountmax2_14A__sum",
"overdueamountmax2_398A__mean",
"overdueamountmax2_398A__max",
"overdueamountmax2_398A__std",
"prolongationcount_599L__sum",
"totalamount_6A__null_rate",
"totalamount_996A__max",
"totaloutstanddebtvalue_39A__sum",
"pmts_dpd_1073P_mean__mean",
"pmts_dpd_1073P_mean__max",
"pmts_dpd_1073P_max__max",
"pmts_dpd_1073P_n_unique__max",
"pmts_dpd_1073P_positive_count__sum",
"pmts_dpd_1073P_positive_count__max",
"pmts_dpd_1073P_mean_positive__max",
"pmts_dpd_1073P_sum_positive__sum",
"pmts_dpd_1073P_sum_positive__max",
"pmts_dpd_1073P_non_null_count__sum",
"pmts_dpd_303P_mean__max",
"pmts_dpd_303P_max__max",
"pmtamount_36A_non_null_count",
"numberofqueries_373L",
"b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed",
"b2_pmts_pmtsoverdue_635A_mean_weighted",
"b2_pmts_pmtsoverdue_635A_mean_contract_mean",
"b2_pmts_pmtsoverdue_635A_overdue_rate_contract_max",
"b2_pmts_pmtsoverdue_635A_mean_contract_max",
"b2_pmts_dpdvalue_108P_mean_contract_mean",
"pmts_overdue_1140A_non_null_count__sum",
"pmts_overdue_1152A_mean__mean",
"pmts_overdue_1152A_std__mean",
"pmts_overdue_1140A_mean__max",
"pmts_dpd_1073P_recent12_mean__mean",
"pmts_dpd_303P_recent12_mean__max",
"pmts_dpd_1073P_recent12_mean__max",
"pmts_overdue_1152A_median__mean",
"pctinstlsallpaidlate6d_3546844L",
"pmts_dpd_303P_median__mean",
"datelastunpaid_3546854D",
"prolongationcount_599L__mean",
])

lf1.sink_parquet("base_final3.parquet")


# %%
#region[rgba(230,126,34,0.15)]

import polars as pl
import lightgbm as lgb
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    log_loss,
    brier_score_loss
)

train = pl.scan_parquet("train_base_final3.parquet")
test = pl.scan_parquet("test_base_final3.parquet")



# x_train = train.drop(["case_id","target","WEEK_NUM"])
y_train = train.select("target")
# x_test = test.drop(["case_id","target","WEEK_NUM"])
y_test = test.select("target")

## 第一版 拔掉三個日期
# x_train = train.drop(["case_id","target","WEEK_NUM","lastrejectdate_50D","maxdpdinstldate_3546855D","lastdelinqdate_224D"])
# x_test = test.drop(["case_id","target","WEEK_NUM","lastrejectdate_50D","maxdpdinstldate_3546855D","lastdelinqdate_224D"])

## 第二版 剩餘60特徵
# x_train = train.select([
#     "registaddr_zipcode_184M_appl",
# "price_1097A",
# "age_years_appl",
# "disbursedcredamount_1113A",
# "interestrate_311L",
# "amount_4527230A_sum_positive",
# "residualamount_856A__min",
# "pmts_overdue_1140A_overdue_rate__weighted_avg",
# "totalamount_996A__mean",
# "numberofoutstandinstls_59L__min",
# "pmts_dpd_1073P_recent_time_key__mean_fallback",
# "tenure_years_max",
# "totalamount_6A__sum",
# "overdueamountmax_155A__std",
# "numberofoverdueinstlmax_1039L__min",
# "numinstunpaidmaxest_4493212L",
# "pctinstlsallpaidlate1d_3546856L",
# "pmts_dpd_303P_trend__mean",
# "pmts_overdue_1140A_non_null_count__max",
# "maxdpdinstldate_3546855D_days",
# "numberofoutstandinstls_59L__sum",
# "overdueamountmax_35A__mean",
# "cntpmts24_3658933L",
# "maxdbddpdtollast12m_3658940P",
# "pmtaverage_3A",
# "lastrejectdate_50D_days",
# "incometype_1044T_appl",
# "pctinstlsallpaidearl3d_427L",
# "reject_rate",
# "pmts_dpd_303P_longest_good_streak__max",
# "pmts_dpd_303P_mean_positive__recomputed",
# "lastdelinqdate_224D_days",
# "overdueamountmax_35A__std",
# "days360_512L",
# "tenure_years_appl",
# "pmts_dpd_303P_std__max",
# "totalsettled_863A",
# "monthlyinstlamount_674A__mean",
# "pmts_dpd_303P_recent_time_key__mean_fallback",
# "maxdpdtolerance_577P_mean",
# "days90_310L",
# "outstandingamount_362A__max",
# "pmts_dpd_303P_non_null_count__sum",
# "pmts_dpd_1073P_trend__max",
# "numinstlswithoutdpd_562L",
# "maxdebt4_972A",
# "pmts_dpd_1073P_trend__mean",
# "maxannuity_4075009A",
# "numinstlswithdpd10_728L",
# "disbursementtype_67L",
# "maxdpdlast3m_392P",
# "days180_256L",
# "prolongationcount_599L__null_rate",
# "rejectreasonclient_4145042M_non_placeholder_ratio",
# "amount_4527230A_positive_count",
# "last_status",
# "familystate_447L_appl",
# "pmts_dpd_303P_recent_time_key__max_fallback",
# "subjectroles_name_838M_entropy__mean",
# "avgdpdtolclosure24_3658938P"
# ])
# x_test = test.select([
#     "registaddr_zipcode_184M_appl",
# "price_1097A",
# "age_years_appl",
# "disbursedcredamount_1113A",
# "interestrate_311L",
# "amount_4527230A_sum_positive",
# "residualamount_856A__min",
# "pmts_overdue_1140A_overdue_rate__weighted_avg",
# "totalamount_996A__mean",
# "numberofoutstandinstls_59L__min",
# "pmts_dpd_1073P_recent_time_key__mean_fallback",
# "tenure_years_max",
# "totalamount_6A__sum",
# "overdueamountmax_155A__std",
# "numberofoverdueinstlmax_1039L__min",
# "numinstunpaidmaxest_4493212L",
# "pctinstlsallpaidlate1d_3546856L",
# "pmts_dpd_303P_trend__mean",
# "pmts_overdue_1140A_non_null_count__max",
# "maxdpdinstldate_3546855D_days",
# "numberofoutstandinstls_59L__sum",
# "overdueamountmax_35A__mean",
# "cntpmts24_3658933L",
# "maxdbddpdtollast12m_3658940P",
# "pmtaverage_3A",
# "lastrejectdate_50D_days",
# "incometype_1044T_appl",
# "pctinstlsallpaidearl3d_427L",
# "reject_rate",
# "pmts_dpd_303P_longest_good_streak__max",
# "pmts_dpd_303P_mean_positive__recomputed",
# "lastdelinqdate_224D_days",
# "overdueamountmax_35A__std",
# "days360_512L",
# "tenure_years_appl",
# "pmts_dpd_303P_std__max",
# "totalsettled_863A",
# "monthlyinstlamount_674A__mean",
# "pmts_dpd_303P_recent_time_key__mean_fallback",
# "maxdpdtolerance_577P_mean",
# "days90_310L",
# "outstandingamount_362A__max",
# "pmts_dpd_303P_non_null_count__sum",
# "pmts_dpd_1073P_trend__max",
# "numinstlswithoutdpd_562L",
# "maxdebt4_972A",
# "pmts_dpd_1073P_trend__mean",
# "maxannuity_4075009A",
# "numinstlswithdpd10_728L",
# "disbursementtype_67L",
# "maxdpdlast3m_392P",
# "days180_256L",
# "prolongationcount_599L__null_rate",
# "rejectreasonclient_4145042M_non_placeholder_ratio",
# "amount_4527230A_positive_count",
# "last_status",
# "familystate_447L_appl",
# "pmts_dpd_303P_recent_time_key__max_fallback",
# "subjectroles_name_838M_entropy__mean",
# "avgdpdtolclosure24_3658938P"
# ])

## 第三版 剩餘30特徵

x_train = train.select(["registaddr_zipcode_184M_appl",
"price_1097A",
"age_years_appl",
"amount_4527230A_sum_positive",
"disbursedcredamount_1113A",
"pmts_dpd_1073P_recent_time_key__mean_fallback",
"interestrate_311L",
"residualamount_856A__min",
"pmts_overdue_1140A_overdue_rate__weighted_avg",
"totalamount_996A__mean",
"tenure_years_max",
"pmts_overdue_1140A_non_null_count__max",
"numberofoutstandinstls_59L__min",
"totalamount_6A__sum",
"pmts_dpd_303P_trend__mean",
"overdueamountmax_155A__std",
"numberofoutstandinstls_59L__sum",
"maxdbddpdtollast12m_3658940P",
"cntpmts24_3658933L",
"lastrejectdate_50D_days",
"numberofoverdueinstlmax_1039L__min",
"pmts_dpd_303P_recent_time_key__mean_fallback",
"pmts_dpd_303P_mean_positive__recomputed",
"maxdpdinstldate_3546855D_days",
"numinstunpaidmaxest_4493212L",
"maxdebt4_972A",
"pmts_dpd_303P_non_null_count__sum",
"incometype_1044T_appl",
"overdueamountmax_35A__mean",
"pctinstlsallpaidearl3d_427L"
])
x_test = test.select(["registaddr_zipcode_184M_appl",
"price_1097A",
"age_years_appl",
"amount_4527230A_sum_positive",
"disbursedcredamount_1113A",
"pmts_dpd_1073P_recent_time_key__mean_fallback",
"interestrate_311L",
"residualamount_856A__min",
"pmts_overdue_1140A_overdue_rate__weighted_avg",
"totalamount_996A__mean",
"tenure_years_max",
"pmts_overdue_1140A_non_null_count__max",
"numberofoutstandinstls_59L__min",
"totalamount_6A__sum",
"pmts_dpd_303P_trend__mean",
"overdueamountmax_155A__std",
"numberofoutstandinstls_59L__sum",
"maxdbddpdtollast12m_3658940P",
"cntpmts24_3658933L",
"lastrejectdate_50D_days",
"numberofoverdueinstlmax_1039L__min",
"pmts_dpd_303P_recent_time_key__mean_fallback",
"pmts_dpd_303P_mean_positive__recomputed",
"maxdpdinstldate_3546855D_days",
"numinstunpaidmaxest_4493212L",
"maxdebt4_972A",
"pmts_dpd_303P_non_null_count__sum",
"incometype_1044T_appl",
"overdueamountmax_35A__mean",
"pctinstlsallpaidearl3d_427L"
])



week_train_pd = (
    train
    .select("WEEK_NUM")
    .collect()
    .to_numpy()
    .ravel()
)

week_test_pd = (
    test
    .select("WEEK_NUM")
    .collect()
    .to_numpy()
    .ravel()
)


x_train_pd = x_train.collect().to_pandas()
y_train_pd = y_train.collect().to_numpy().ravel()
x_test_pd = x_test.collect().to_pandas()
y_test_pd = y_test.collect().to_numpy().ravel()


categorical_cols = list(
    x_train_pd.select_dtypes(include=["object"]).columns
)

for col in categorical_cols:
    x_train_pd[col] = x_train_pd[col].astype("category")
    x_test_pd[col] = x_test_pd[col].astype("category")


# x_train_pd.shape
# y_train_pd.shape


model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=31,

    random_state=42,

    n_jobs=-1
)

model.fit(
    x_train_pd,
    y_train_pd
)

train_probability = model.predict_proba(x_train_pd)[:,1]
test_probability = model.predict_proba(x_test_pd)[:,1]

def evaluate_model(
    y_true,
    probability,
    dataset_name
):

    auc = roc_auc_score(
        y_true,
        probability
    )

    gini = 2*auc-1

    fpr,tpr,_ = roc_curve(
        y_true,
        probability
    )

    ks = np.max(
        tpr-fpr
    )

    ll = log_loss(
        y_true,
        probability
    )

    brier = brier_score_loss(
        y_true,
        probability
    )

    print("="*40)
    print(dataset_name)
    print("="*40)

    print(f"AUC         : {auc:.6f}")
    print(f"Gini        : {gini:.6f}")
    print(f"KS          : {ks:.6f}")
    print(f"LogLoss     : {ll:.6f}")
    print(f"Brier Score : {brier:.6f}")

def evaluate_home_credit_official(
    y_true,
    probability,
    week_num,
    dataset_name
):
    y_true = np.asarray(y_true).ravel()
    probability = np.asarray(probability).ravel()
    week_num = np.asarray(week_num).ravel()

    if not (
        len(y_true)
        == len(probability)
        == len(week_num)
    ):
        raise ValueError(
            "y_true、probability、week_num 的資料筆數必須相同。"
        )

    if np.isnan(probability).any():
        raise ValueError(
            "預測機率中存在 NaN。"
        )

    if (
        (probability < 0)
        | (probability > 1)
    ).any():
        raise ValueError(
            "預測機率必須介於 0 和 1 之間。"
        )

    evaluation_df = pd.DataFrame({
        "target": y_true,
        "probability": probability,
        "WEEK_NUM": week_num
    })

    weekly_results = []

    for week, week_df in evaluation_df.groupby(
        "WEEK_NUM",
        sort=True
    ):
        week_target = week_df["target"]
        week_probability = week_df["probability"]

        if week_target.nunique() < 2:
            print(
                f"警告：WEEK_NUM={week} 只有單一 target，"
                "本週略過，不計算 Gini。"
            )
            continue

        week_auc = roc_auc_score(
            week_target,
            week_probability
        )

        week_gini = 2 * week_auc - 1

        weekly_results.append({
            "WEEK_NUM": week,
            "sample_size": len(week_df),
            "bad_count": int(week_target.sum()),
            "bad_rate": float(
                week_target.mean()
            ),
            "auc": week_auc,
            "gini": week_gini
        })

    weekly_df = (
        pd.DataFrame(weekly_results)
        .sort_values("WEEK_NUM")
        .reset_index(drop=True)
    )

    if len(weekly_df) < 2:
        raise ValueError(
            "至少需要兩個有效 WEEK_NUM，"
            "才能計算官方 Stability Metric。"
        )

    x = weekly_df[
        "WEEK_NUM"
    ].to_numpy(dtype=float)

    y = weekly_df[
        "gini"
    ].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        deg=1
    )

    fitted_gini = (
        slope * x
        + intercept
    )

    residuals = (
        y
        - fitted_gini
    )

    mean_gini = np.mean(y)

    falling_rate = min(
        0.0,
        slope
    )

    falling_penalty = (
        88.0
        * falling_rate
    )

    residual_std = np.std(
        residuals,
        ddof=0
    )

    variability_penalty = (
        0.5
        * residual_std
    )

    stability_metric = (
        mean_gini
        + falling_penalty
        - variability_penalty
    )

    summary_df = pd.DataFrame({
    "Dataset": [dataset_name] * 8,
    "Metric": [
        "Valid Weeks",
        "Mean Weekly Gini",
        "Gini Trend Slope",
        "Falling Rate",
        "Falling Penalty",
        "Residual STD",
        "Variability Penalty",
        "Official Stability"
    ],
    "Value": [
        len(weekly_df),
        mean_gini,
        slope,
        falling_rate,
        falling_penalty,
        residual_std,
        variability_penalty,
        stability_metric
    ]
})

    return {
        "stability_metric": stability_metric,
        "mean_weekly_gini": mean_gini,
        "slope": slope,
        "falling_rate": falling_rate,
        "falling_penalty": falling_penalty,
        "residual_std": residual_std,
        "variability_penalty": variability_penalty,
        "summary": summary_df,
        "weekly_results": weekly_df
    }




# evaluate_model(
#     y_train_pd,
#     train_probability,
#     "Train"
# )

# evaluate_model(
#     y_test_pd,
#     test_probability,
#     "Test"
# )

train_official_result = evaluate_home_credit_official(
    y_true=y_train_pd,
    probability=train_probability,
    week_num=week_train_pd,
    dataset_name="Train"
)

test_official_result = evaluate_home_credit_official(
    y_true=y_test_pd,
    probability=test_probability,
    week_num=week_test_pd,
    dataset_name="Test"
)

## 合併成 Train/Test 橫向比較表

official_comparison = pd.concat(
    [
        train_official_result["summary"],
        test_official_result["summary"]
    ],
    ignore_index=True
)

official_comparison_table = (
    official_comparison
    .pivot(
        index="Metric",
        columns="Dataset",
        values="Value"
    )
    .reindex([
        "Valid Weeks",
        "Mean Weekly Gini",
        "Gini Trend Slope",
        "Falling Rate",
        "Falling Penalty",
        "Residual STD",
        "Variability Penalty",
        "Official Stability"
    ])
    .reset_index()
)

official_comparison_table.columns.name = None

# 顯示表格

display(
    official_comparison_table.style
    .format({
        "Train": "{:.6f}",
        "Test": "{:.6f}"
    })
    .hide(axis="index")
)


#endregion
# %%

### 畫重要度圖

importance_df = pd.DataFrame({
    "Feature": x_train_pd.columns,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

display(importance_df)

import matplotlib.pyplot as plt
import lightgbm as lgb

plt.figure(figsize=(10,12))

lgb.plot_importance(
    model,
    max_num_features=30,
    importance_type="gain"
)

plt.show()

importance_df.to_csv("importance_df.csv", index=False, encoding="utf-8-sig")


# %%

### 處理三個日期格式(使用申請日相減)

a = pl.scan_parquet("train_base_final.parquet")
# b = pl.scan_parquet("test_base_final.parquet")
c = pl.scan_parquet("train_base.parquet")
c1 = c.select(["case_id","date_decision"])

a1 = a.join(c1, on="case_id", how="left")
# b1 = b.join(c1, on="case_id", how="left")

a2 = a1.with_columns(
    lastrejectdate_50D_days = (pl.col("date_decision").str.to_date() - pl.col("lastrejectdate_50D").str.to_date()).dt.total_days(),
    maxdpdinstldate_3546855D_days = (pl.col("date_decision").str.to_date() - pl.col("maxdpdinstldate_3546855D").str.to_date()).dt.total_days(),
    lastdelinqdate_224D_days = (pl.col("date_decision").str.to_date() - pl.col("lastdelinqdate_224D").str.to_date()).dt.total_days()
)
# b2 = b1.with_columns(
#     lastrejectdate_50D_days = (pl.col("date_decision").str.to_date() - pl.col("lastrejectdate_50D").str.to_date()).dt.total_days(),
#     maxdpdinstldate_3546855D_days = (pl.col("date_decision").str.to_date() - pl.col("maxdpdinstldate_3546855D").str.to_date()).dt.total_days(),
#     lastdelinqdate_224D_days = (pl.col("date_decision").str.to_date() - pl.col("lastdelinqdate_224D").str.to_date()).dt.total_days()
# )

#lastdelinqdate_224D_days
a3 = a2.with_columns(
    pl.when(pl.col("lastdelinqdate_224D_days") < 0)
    .then(None)
    .otherwise(pl.col("lastdelinqdate_224D_days"))
    .alias("lastdelinqdate_224D_days")
)
# b3 = b2.with_columns(
#     pl.when(pl.col("lastdelinqdate_224D_days") < 0)
#     .then(None)
#     .otherwise(pl.col("lastdelinqdate_224D_days"))
#     .alias("lastdelinqdate_224D_days")
# )

# #maxdpdinstldate_3546855D_days
a4 = a3.with_columns(
    pl.when(pl.col("maxdpdinstldate_3546855D_days") < 0)
    .then(None)
    .otherwise(pl.col("maxdpdinstldate_3546855D_days"))
    .alias("maxdpdinstldate_3546855D_days")
)
# b4 = b3.with_columns(
#     pl.when(pl.col("maxdpdinstldate_3546855D_days") < 0)
#     .then(None)
#     .otherwise(pl.col("maxdpdinstldate_3546855D_days"))
#     .alias("maxdpdinstldate_3546855D_days")
# )

# #lastrejectdate_50D_days
a5 = a4.with_columns(
    pl.when(pl.col("lastrejectdate_50D_days") < 0)
    .then(None)
    .otherwise(pl.col("lastrejectdate_50D_days"))
    .alias("lastrejectdate_50D_days")
)
# b5 = b4.with_columns(
#     pl.when(pl.col("lastrejectdate_50D_days") < 0)
#     .then(None)
#     .otherwise(pl.col("lastrejectdate_50D_days"))
#     .alias("lastrejectdate_50D_days")
# )

# # 查看負值比例
# negative_rate_value = (
#     a3.select(
#         (pl.col("lastdelinqdate_224D_days") < 0)
#         .mean()
#         .alias("negative_rate")
#     )
#     .collect()
#     .item()
# )

# print(negative_rate_value)



a5.sink_parquet("train_base_final2.parquet")
# b5.sink_parquet("test_base_final2.parquet")

# %%

a = pl.scan_parquet("train_base_final2.parquet")
b = pl.scan_parquet("test_base_final2.parquet")

a = a.drop("date_decision")
b = b.drop("date_decision")

a.sink_parquet("train_base_final3.parquet")
b.sink_parquet("test_base_final3.parquet")
# %%
