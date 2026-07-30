"""建立 train/test 建模資料集(依 build_model_dataset_plan.md)。

本檔以【資料工程 DATA ENGINEERING】與【特徵工程 FEATURE ENGINEERING】兩大區塊組織:

【資料工程】把原始分區檔整理成乾淨、正確型別的表:
  1. union 各表分區檔(train_.../test_...)。
  2. 依欄位大寫標籤標準化型別(transform_by_label):
     P=DPD 天數→數值、A=金額→數值、D=日期→Date、M=遮罩類別→字串、T/L=未指定維持原生型別。
  3. date_decision 減 3 個日期欄 → _days 欄(add_date_diff_days)。

【特徵工程】從乾淨表建構模型特徵:
  4. 合理值清洗:連續型 99%PR 截尾(p99==0 不補)、A 欄負值→null、_days 欄負值→null、原 D 欄移除。
  5. 依後贅詞聚合(depth-0 直接 join、depth-1 一次聚合、depth-2 兩次聚合)。
  6. 7 個財務基礎欄聚合 + 3 個財務比率特徵(dti_ratio / overdue_debt_ratio / deposit_to_debt_ratio)。
  7. 5 個衍生欄(reject_rate/last_status/tenure_years_max/age_years_appl/tenure_years_appl)。
  8. join 回 base,輸出 df_train / df_test(Polars DataFrame + parquet)。

credit_bureau_a_2(depth-2)資料量達 1.88 億列,使用 DuckDB 聚合(regr_slope 等視窗/聚合函式
對超大表更穩健);其餘表用 Polars(沿用既有 data_preprocessing_2.py / train_person_agg.py 風格)。
credit_bureau_b(b1/b2)依使用者指示忽略,不納入。
"""


import glob
import math
import os

import duckdb
import polars as pl
import lightgbm as lgb
from pathlib import Path
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


ROOT            = Path("/kaggle/input/competitions/home-credit-credit-risk-model-stability")
TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

M_PLACEHOLDER = "a55475b1"
REFERENCE_YEAR = 2024
INVALID_YEAR_LO, INVALID_YEAR_HI = 2025, 2028  # bureau_a2_merge_GroupRule.txt 規則

# #############################################################################
# 【資料工程 DATA ENGINEERING】union 分區 → 標籤型別標準化 → 日期轉 days
# #############################################################################

def read_table(data_dir: str, split: str, table: str) -> pl.LazyFrame:
    """[資料工程] 讀取一張表,若有分區檔(depth 後贅詞_N)則 union,否則讀單檔。"""
    exact = f"{data_dir}/{split}_{table}.parquet"
    if os.path.exists(exact):
        return pl.scan_parquet(exact)
    pattern = f"{data_dir}/{split}_{table}_*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched {exact} or {pattern}")
    return pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")


def raw_glob(data_dir: str, split: str, table: str) -> str:
    """[資料工程] 回傳供 DuckDB read_parquet 用的 glob pattern(單檔或多分區)。"""
    exact = f"{data_dir}/{split}_{table}.parquet"
    if os.path.exists(exact):
        return exact
    return f"{data_dir}/{split}_{table}_*.parquet"


def transform_by_label(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """[資料工程] 依欄位大寫尾碼做格式(型別)標準化:
    D→Date(Transform date)、A→Float64(Transform amount)、P→Float64(Transform DPD)、
    M→Utf8(Masking categories,維持遮罩雜湊字串)、T/L→維持原生型別(Unspecified)。
    """
    exprs = []
    for c in cols:
        if c not in df.columns:
            continue
        dtype = df.schema[c]
        if c.endswith("D"):
            if dtype == pl.Utf8:  # 日期字串才 parse;已是 Date 則略過
                exprs.append(pl.col(c).str.strptime(pl.Date, strict=False).alias(c))
        elif c.endswith("A") or c.endswith("P"):
            if dtype != pl.Float64:
                exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        elif c.endswith("M"):
            if dtype != pl.Utf8:
                exprs.append(pl.col(c).cast(pl.Utf8).alias(c))
        # T / L:未指定 Transform,維持原生型別
    return df.with_columns(exprs) if exprs else df


def add_date_diff_days(
    df: pl.DataFrame, date_cols: list[str], ref_col: str = "date_decision"
) -> pl.DataFrame:
    """[資料工程] {col}_days = (date_decision − col) 的天數;原 D 欄保留待特徵工程階段移除。
    ref_col 與 date_cols 皆須為 pl.Date。"""
    exprs = [
        (pl.col(ref_col) - pl.col(c)).dt.total_days().alias(f"{c}_days")
        for c in date_cols
        if c in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


# #############################################################################
# 【特徵工程 FEATURE ENGINEERING】清洗 → 聚合 → 財務指標 → 衍生欄 → join
# #############################################################################

def winsorize_p99(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """[特徵工程] 連續型 99%PR 截尾(不刪列);若 p99 為 0 或 null 則不截尾。"""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    p99 = df.select([pl.col(c).quantile(0.99).alias(c) for c in cols]).row(0)
    exprs = []
    for c, cap in zip(cols, p99):
        if cap is None or abs(cap) < 1e-9:  # 視為 0(容忍近似分位數的浮點雜訊)
            continue
        exprs.append(pl.when(pl.col(c) > cap).then(cap).otherwise(pl.col(c)).alias(c))
    return df.with_columns(exprs) if exprs else df


def clean_by_label(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """[特徵工程] A 尾碼(金額)負值 -> null;_days 尾碼(日期差天數)負值 -> null。"""
    exprs = [
        pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
        if c in df.columns and (c.endswith("A") or c.endswith("_days"))
    ]
    return df.with_columns(exprs) if exprs else df


def clean_future_d_cols(df: pl.DataFrame, cols: list[str], ref_col: str) -> pl.DataFrame:
    """[特徵工程] 大寫 D 結尾(日期)欄位晚於 ref_col 者 -> null(內部衍生用,兩者皆須為 pl.Date)。"""
    d_cols = [c for c in cols if c.endswith("D") and c in df.columns]
    if not d_cols:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(c) > pl.col(ref_col)).then(None).otherwise(pl.col(c)).alias(c)
            for c in d_cols
        ]
    )


def cat_group_stats(
    df: pl.DataFrame, group_keys: list[str], col: str, stats: tuple[str, ...]
) -> pl.DataFrame:
    """類別欄分組統計:mode/n_unique/entropy/non_placeholder_ratio(沿用既有 build_cat_agg 慣例)。"""
    vc = df.drop_nulls(col).group_by(group_keys + [col]).agg(pl.len().alias("cnt"))
    exprs = []
    if "mode" in stats:
        exprs.append(
            pl.col(col)
            .sort_by(["cnt", col], descending=[True, False])
            .first()
            .alias(f"{col}_mode")
        )
    if "n_unique" in stats:
        exprs.append(pl.len().alias(f"{col}_n_unique"))
    if "entropy" in stats:
        exprs.append(pl.col("cnt").entropy(base=math.e, normalize=True).alias(f"{col}_entropy"))
    if "non_placeholder_ratio" in stats:
        exprs.append(
            (pl.col("cnt").filter(pl.col(col) != M_PLACEHOLDER).sum() / pl.col("cnt").sum())
            .fill_null(0.0)
            .alias(f"{col}_non_placeholder_ratio")
        )
    return vc.group_by(group_keys).agg(exprs)


NUMERIC_STAT_FUNCS = {
    "mean": lambda c: c.mean(),
    "max": lambda c: c.max(),
    "min": lambda c: c.min(),
    "median": lambda c: c.median(),
    "std": lambda c: c.std(),
    "sum": lambda c: c.sum(),
    "null_rate": lambda c: c.is_null().mean(),
    "null_count": lambda c: c.is_null().sum(),
    "non_null_count": lambda c: c.is_not_null().sum(),
    "n_unique": lambda c: c.n_unique(),
    "positive_count": lambda c: (c > 0).sum(),
    "overdue_rate": lambda c: (c > 0).sum() / c.is_not_null().sum(),
    "mean_positive": lambda c: pl.when(c > 0).then(c).otherwise(None).mean(),
    "sum_positive": lambda c: pl.when(c > 0).then(c).otherwise(0).sum(),
}


def stat_expr(raw: str, stat: str) -> pl.Expr:
    return NUMERIC_STAT_FUNCS[stat](pl.col(raw)).alias(f"{raw}__{stat}")


def financial_stat_expr(raw: str, stat: str) -> pl.Expr:
    """財務基礎欄聚合:輸出沿用原始欄名(代表 case 層級值),不加 __suffix。"""
    return NUMERIC_STAT_FUNCS[stat](pl.col(raw)).alias(raw)


# =============================================================================
# depth-0:static_0 + static_cb_0(直接 join,無需聚合)
# =============================================================================

# 財務基礎欄(見 build_model_dataset_plan.md F3):totaldebt_9A、maininc_215A 位於 static_0
STATIC_0_COLS = [
    "avgdpdtolclosure24_3658938P", "maxdbddpdtollast12m_3658940P", "pctinstlsallpaidlate1d_3546856L",
    "pctinstlsallpaidearl3d_427L", "lastrejectreason_759M", "cntpmts24_3658933L",
    "disbursedcredamount_1113A", "maxdpdlast12m_727P", "maxdpdlast24m_143P", "numrejects9m_859L",
    "price_1097A", "interestrate_311L", "maxdebt4_972A", "maxdpdtolerance_374P",
    "numinstlswithdpd10_728L", "numinstlswithdpd5_4187116L", "totalsettled_863A",
    "disbursementtype_67L", "maxdpdlast3m_392P", "numinstlswithoutdpd_562L",
    "numinstunpaidmaxest_4493212L", "maxannuity_4075009A", "credtype_322L", "opencred_647L",
    "lastrejectdate_50D", "maxdpdinstldate_3546855D", "lastdelinqdate_224D",
    "totaldebt_9A", "maininc_215A",  # 財務基礎欄(直取,1:1)
]
# date_decision 減這些 D 欄 → _days,原 D 欄之後移除(需求 4/5)
STATIC_0_DATE_TO_DAYS = ["lastrejectdate_50D", "maxdpdinstldate_3546855D", "lastdelinqdate_224D"]
STATIC_CB_0_COLS = [
    "education_1103M", "pmtaverage_3A", "requesttype_4525192L", "days180_256L",
    "days30_165L", "days360_512L", "days90_310L", "riskassesment_940T",
]


def build_static(data_dir: str, split: str, base_dates: pl.DataFrame) -> pl.DataFrame:
    s0 = read_table(data_dir, split, "static_0").select(["case_id"] + STATIC_0_COLS).collect()
    scb0 = read_table(data_dir, split, "static_cb_0").select(["case_id"] + STATIC_CB_0_COLS).collect()

    # [資料工程] 標籤型別標準化(D→Date、A→Float64、M→Utf8...)
    s0 = transform_by_label(s0, STATIC_0_COLS)
    scb0 = transform_by_label(scb0, STATIC_CB_0_COLS)

    # [資料工程] date_decision − D → _days
    s0 = s0.join(base_dates, on="case_id", how="left")
    s0 = add_date_diff_days(s0, STATIC_0_DATE_TO_DAYS)

    # [特徵工程] 清洗:A 負值/_days 負值 → null;連續型(含 _days)99%PR 截尾
    days_cols_0 = [f"{c}_days" for c in STATIC_0_DATE_TO_DAYS]
    s0 = clean_by_label(s0, STATIC_0_COLS + days_cols_0)
    numeric_0 = [c for c in STATIC_0_COLS if s0.schema[c].is_numeric()] + days_cols_0
    s0 = winsorize_p99(s0, numeric_0)
    # [特徵工程] 移除原 D 欄(已轉 _days)與暫用的 date_decision
    s0 = s0.drop(STATIC_0_DATE_TO_DAYS + ["date_decision"])

    numeric_cb0 = [c for c in STATIC_CB_0_COLS if scb0.schema[c].is_numeric()]
    scb0 = clean_by_label(scb0, STATIC_CB_0_COLS)
    scb0 = winsorize_p99(scb0, numeric_cb0)

    return s0.join(scb0, on="case_id", how="left")


# =============================================================================
# depth-1:credit_bureau_a_1(單層後贅詞統計)
# =============================================================================

A1_SPEC = [
    ("numberofoutstandinstls_59L", "min"),
    ("numberofoutstandinstls_59L", "sum"),
    ("overdueamountmax_35A", "mean"),
    ("overdueamountmax_35A", "std"),
    ("numberofoverdueinstlmax_1039L", "min"),
    ("prolongationcount_599L", "null_rate"),
    ("residualamount_856A", "min"),
    ("totalamount_6A", "sum"),
    ("totalamount_996A", "mean"),
    ("dpdmax_757P", "null_rate"),
    ("monthlyinstlamount_674A", "mean"),
    ("outstandingamount_362A", "max"),
    ("overdueamountmax_155A", "std"),
]
# 財務基礎欄(F3):存續合約逾期/未償債務,跨合約 sum,輸出用原始欄名
A1_FINANCIAL = [
    ("totaldebtoverduevalue_178A", "sum"),
    ("totaloutstanddebtvalue_39A", "sum"),
]


def build_bureau_a1(data_dir: str, split: str) -> pl.DataFrame:
    raw_cols = sorted({raw for raw, _ in A1_SPEC} | {raw for raw, _ in A1_FINANCIAL})
    df = read_table(data_dir, split, "credit_bureau_a_1").select(["case_id"] + raw_cols).collect()
    # [資料工程] 標籤型別標準化
    df = transform_by_label(df, raw_cols)
    # [特徵工程] 清洗 + 截尾
    df = clean_by_label(df, raw_cols)
    numeric_cols = [c for c in raw_cols if df.schema[c].is_numeric()]
    df = winsorize_p99(df, numeric_cols)
    # [特徵工程] depth-1 一次聚合 by case_id(既有後贅詞統計 + 財務欄 sum)
    exprs = [stat_expr(raw, stat) for raw, stat in A1_SPEC]
    exprs += [financial_stat_expr(raw, stat) for raw, stat in A1_FINANCIAL]
    return df.group_by("case_id").agg(exprs)


# =============================================================================
# depth-1:applprev_1(直接欄 mean + 類別 mode/non_placeholder_ratio + 5 衍生欄)
# =============================================================================

def build_applprev(data_dir: str, split: str) -> pl.DataFrame:
    cols = [
        "case_id", "num_group1", "status_219L", "creationdate_885D", "employedfrom_700D",
        "maxdpdtolerance_577P", "rejectreasonclient_4145042M", "cancelreason_3545846M",
        "education_1138M",
    ]
    df = read_table(data_dir, split, "applprev_1").select(cols).collect()

    df = df.with_columns(
        pl.col("creationdate_885D").str.strptime(pl.Date, strict=False),
        pl.col("employedfrom_700D").str.strptime(pl.Date, strict=False),
    )
    df = clean_future_d_cols(df, ["employedfrom_700D"], "creationdate_885D")
    df = winsorize_p99(df, ["maxdpdtolerance_577P"])

    df = df.with_columns(
        (
            REFERENCE_YEAR
            - (
                pl.col("employedfrom_700D").dt.year()
                + (pl.col("employedfrom_700D").dt.month() - 1) / 12
            )
        ).alias("tenure_years")
    )
    df = df.sort(["case_id", "creationdate_885D"])

    numeric_agg = (
        df.group_by("case_id")
        .agg(
            pl.col("maxdpdtolerance_577P").mean().alias("maxdpdtolerance_577P_mean"),
            (pl.col("status_219L") == "D").sum().alias("_n_rejected"),
            pl.col("status_219L").is_not_null().sum().alias("_n_status_valid"),
            pl.col("status_219L").last().alias("last_status"),
            pl.col("tenure_years").max().alias("tenure_years_max"),
        )
        .with_columns(
            (pl.col("_n_rejected") / pl.col("_n_status_valid")).alias("reject_rate")
        )
        .drop(["_n_rejected", "_n_status_valid"])
    )

    cancel_stats = cat_group_stats(df, ["case_id"], "cancelreason_3545846M", ("mode",))
    edu_stats = cat_group_stats(df, ["case_id"], "education_1138M", ("mode",))
    reject_stats = cat_group_stats(
        df, ["case_id"], "rejectreasonclient_4145042M", ("non_placeholder_ratio",)
    )

    return (
        numeric_agg.join(cancel_stats, on="case_id", how="left")
        .join(edu_stats, on="case_id", how="left")
        .join(reject_stats, on="case_id", how="left")
    )


# =============================================================================
# depth-1:tax_registry_a_1
# =============================================================================

def build_tax_a(data_dir: str, split: str) -> pl.DataFrame:
    df = read_table(data_dir, split, "tax_registry_a_1").select(["case_id", "amount_4527230A"]).collect()
    # [資料工程] 標籤型別標準化 → [特徵工程] 清洗 + 截尾 + 聚合
    df = transform_by_label(df, ["amount_4527230A"])
    df = clean_by_label(df, ["amount_4527230A"])
    df = winsorize_p99(df, ["amount_4527230A"])
    return df.group_by("case_id").agg(
        stat_expr("amount_4527230A", "sum_positive").alias("amount_4527230A_sum_positive"),
        stat_expr("amount_4527230A", "positive_count").alias("amount_4527230A_positive_count"),
    )


# =============================================================================
# depth-1:person_1(本人列 _appl 直取 + age/tenure 衍生)
# =============================================================================

def build_person(data_dir: str, split: str, base_dates: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "case_id", "num_group1", "education_927M", "incometype_1044T", "familystate_447L",
        "empl_industry_691L", "registaddr_zipcode_184M", "birth_259D", "birthdate_87D",
        "empl_employedfrom_271D", "mainoccupationinc_384A",  # 財務基礎欄(F3)
    ]
    df = read_table(data_dir, split, "person_1").select(cols).collect()
    # [資料工程] 標籤型別標準化(D→Date、A→Float64、M→Utf8)
    df = transform_by_label(df, cols)
    df = df.join(base_dates, on="case_id", how="left")
    # [特徵工程] 內部衍生用日期未來日 → null;財務欄清洗 + 截尾
    df = clean_future_d_cols(
        df, ["birth_259D", "birthdate_87D", "empl_employedfrom_271D"], "date_decision"
    )
    df = clean_by_label(df, ["mainoccupationinc_384A"])
    df = winsorize_p99(df, ["mainoccupationinc_384A"])
    df = df.with_columns(pl.coalesce(["birth_259D", "birthdate_87D"]).alias("_birth"))
    df = df.with_columns(
        ((pl.col("date_decision") - pl.col("_birth")).dt.total_days() / 365.25).alias(
            "age_years_appl"
        ),
        ((pl.col("date_decision") - pl.col("empl_employedfrom_271D")).dt.total_days() / 365.25).alias(
            "tenure_years_appl"
        ),
    )

    # [特徵工程] 財務欄 mainoccupationinc_384A:case 層級 max(僅本人列有值,max 即取回本人收入)
    occ_income = df.group_by("case_id").agg(
        financial_stat_expr("mainoccupationinc_384A", "max")
    )

    # 本人列(num_group1==0)直取 _appl 類別欄 + age/tenure
    appl = (
        df.filter(pl.col("num_group1") == 0)
        .select(
            "case_id",
            "age_years_appl",
            "tenure_years_appl",
            pl.col("education_927M").alias("education_927M_appl"),
            pl.col("incometype_1044T").alias("incometype_1044T_appl"),
            pl.col("familystate_447L").alias("familystate_447L_appl"),
            pl.col("empl_industry_691L").alias("empl_industry_691L_appl"),
            pl.col("registaddr_zipcode_184M").alias("registaddr_zipcode_184M_appl"),
        )
    )
    return appl.join(occ_income, on="case_id", how="left")


# =============================================================================
# depth-1:other_1(財務基礎欄:存款流入 / 支出流出,case 層級 sum)
# =============================================================================

OTHER_FINANCIAL = [
    ("amtdepositincoming_4809444A", "sum"),
    ("amtdebitoutgoing_4809440A", "sum"),
]


def build_other(data_dir: str, split: str) -> pl.DataFrame:
    raw_cols = [raw for raw, _ in OTHER_FINANCIAL]
    df = read_table(data_dir, split, "other_1").select(["case_id"] + raw_cols).collect()
    # [資料工程] 標籤型別標準化 → [特徵工程] 清洗 + 截尾 + 聚合(sum,other_1 為 1:1)
    df = transform_by_label(df, raw_cols)
    df = clean_by_label(df, raw_cols)
    df = winsorize_p99(df, raw_cols)
    return df.group_by("case_id").agg(
        [financial_stat_expr(raw, stat) for raw, stat in OTHER_FINANCIAL]
    )


# =============================================================================
# 財務比率特徵(需求 9 / plan F4):以聚合後 case 層級欄位計算
# =============================================================================

def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """除零/除 null 保護:分母為 null 或 0 時回傳 null。"""
    return pl.when(denominator.is_null() | (denominator == 0)).then(None).otherwise(
        numerator / denominator
    )


def build_financial_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """[特徵工程] 依 Financial indicators.csv 建構 3 個財務比率(需 df 已含 7 財務基礎欄)。"""
    income = pl.coalesce(["maininc_215A", "mainoccupationinc_384A"])  # 主要收入,職業收入備援
    return df.with_columns(
        _safe_ratio(pl.col("totaldebt_9A"), income).alias("dti_ratio"),
        _safe_ratio(
            pl.col("totaldebtoverduevalue_178A"), pl.col("totaloutstanddebtvalue_39A")
        ).alias("overdue_debt_ratio"),
        _safe_ratio(
            pl.col("amtdepositincoming_4809444A"), pl.col("amtdebitoutgoing_4809440A")
        ).alias("deposit_to_debt_ratio"),
    )


# =============================================================================
# depth-2:credit_bureau_a_2(DuckDB;規模達 1.88 億列)
# =============================================================================

A2_RAW_COLS = [
    "case_id", "num_group1", "num_group2",
    "pmts_dpd_1073P", "pmts_dpd_303P", "pmts_overdue_1140A",
    "collater_valueofguarantee_1124L", "subjectroles_name_838M",
    "pmts_year_1139T", "pmts_month_158T", "pmts_year_507T", "pmts_month_706T",
]
A2_WINSORIZE_COLS = [
    "pmts_dpd_1073P", "pmts_dpd_303P", "pmts_overdue_1140A", "collater_valueofguarantee_1124L",
]


def build_bureau_a2(data_dir: str, split: str) -> pl.DataFrame:
    os.makedirs(".tmp", exist_ok=True)
    keyed_path = f".tmp/a2_keyed_{split}.parquet"
    streak_path = f".tmp/a2_streak_{split}.parquet"
    l1_path = f".tmp/a2_l1_{split}.parquet"

    con = duckdb.connect()
    con.sql("SET memory_limit='8GB'")
    con.sql("SET preserve_insertion_order=false")
    con.sql(f"SET threads TO {max(1, os.cpu_count() // 2)}")
    src = raw_glob(data_dir, split, "credit_bureau_a_2")
    cols_sql = ", ".join(A2_RAW_COLS)

    # --- p99 (approx,大表用近似分位數控制耗時) + 若為 0 則不截尾 ---
    p99_row = con.sql(
        f"""
        SELECT {", ".join(f"approx_quantile({c}, 0.99) AS {c}" for c in A2_WINSORIZE_COLS)}
        FROM (SELECT {cols_sql} FROM read_parquet('{src}'))
        WHERE {" OR ".join(f"{c} IS NOT NULL" for c in A2_WINSORIZE_COLS)}
        """
    ).fetchone()
    caps = dict(zip(A2_WINSORIZE_COLS, p99_row))
    print(f"  a2 p99 caps: {caps}")

    def cap_expr(col: str) -> str:
        cap = caps.get(col)
        if cap is None or abs(cap) < 1e-9:  # 視為 0(容忍 approx_quantile 的浮點雜訊)
            return col
        return f"LEAST({col}, {cap})"

    # --- 清洗 + 時間鍵,單趟寫出 parquet checkpoint(避免後續多階段重複掃 1.88 億列原始表) ---
    con.sql(
        f"""
        COPY (
            SELECT
                case_id, num_group1, num_group2,
                {cap_expr("CASE WHEN pmts_dpd_1073P < 0 THEN NULL ELSE pmts_dpd_1073P END")} AS pmts_dpd_1073P,
                {cap_expr("CASE WHEN pmts_dpd_303P < 0 THEN NULL ELSE pmts_dpd_303P END")} AS pmts_dpd_303P,
                {cap_expr("pmts_overdue_1140A")} AS pmts_overdue_1140A,
                {cap_expr("CASE WHEN collater_valueofguarantee_1124L < 0 THEN NULL ELSE collater_valueofguarantee_1124L END")} AS collater_valueofguarantee_1124L,
                subjectroles_name_838M,
                CASE WHEN pmts_month_158T BETWEEN 1 AND 12
                          AND pmts_year_1139T NOT BETWEEN {INVALID_YEAR_LO} AND {INVALID_YEAR_HI}
                     THEN pmts_year_1139T * 12 + pmts_month_158T END AS active_time_key,
                CASE WHEN pmts_month_706T BETWEEN 1 AND 12
                          AND pmts_year_507T NOT BETWEEN {INVALID_YEAR_LO} AND {INVALID_YEAR_HI}
                     THEN pmts_year_507T * 12 + pmts_month_706T END AS closed_time_key
            FROM (SELECT {cols_sql} FROM read_parquet('{src}'))
        ) TO '{keyed_path}' (FORMAT PARQUET)
        """
    )
    a2_keyed = f"read_parquet('{keyed_path}')"

    # --- gaps-and-islands:pmts_dpd_303P == 0 的最長連續段(依 closed_time_key, num_group2 排序) ---
    # 寫出獨立 checkpoint,避免與後續 L1 GROUP BY 融合成單一巨大查詢計畫導致記憶體暴增
    con.sql(
        f"""
        COPY (
            WITH ordered AS (
                SELECT case_id, num_group1, pmts_dpd_303P,
                    ROW_NUMBER() OVER (
                        PARTITION BY case_id, num_group1
                        ORDER BY closed_time_key NULLS LAST, num_group2
                    ) AS rn
                FROM {a2_keyed}
                WHERE pmts_dpd_303P IS NOT NULL
            ),
            islands AS (
                SELECT case_id, num_group1,
                    rn - ROW_NUMBER() OVER (
                        PARTITION BY case_id, num_group1, (pmts_dpd_303P = 0)
                        ORDER BY rn
                    ) AS grp,
                    pmts_dpd_303P
                FROM ordered
            ),
            streak_len AS (
                SELECT case_id, num_group1, COUNT(*) AS run_len
                FROM islands
                WHERE pmts_dpd_303P = 0
                GROUP BY case_id, num_group1, grp
            )
            SELECT case_id, num_group1, MAX(run_len) AS dpd303_longest_good_streak
            FROM streak_len
            GROUP BY case_id, num_group1
        ) TO '{streak_path}' (FORMAT PARQUET)
        """
    )
    a2_streak_src = f"read_parquet('{streak_path}')"

    # --- L1:一次聚合 by (case_id, num_group1),寫出 checkpoint ---
    con.sql(
        f"""
        COPY (
            SELECT
                k.case_id, k.num_group1,
                regr_slope(pmts_dpd_1073P, active_time_key) AS dpd1073_trend,
                MAX(active_time_key) FILTER (WHERE active_time_key IS NOT NULL) AS dpd1073_recent_time_key,
                regr_slope(pmts_dpd_303P, closed_time_key) AS dpd303_trend,
                SUM(CASE WHEN pmts_dpd_303P > 0 THEN pmts_dpd_303P ELSE 0 END) AS dpd303_sum_positive,
                SUM(CASE WHEN pmts_dpd_303P > 0 THEN 1 ELSE 0 END) AS dpd303_positive_count,
                COUNT(pmts_dpd_303P) AS dpd303_non_null_count,
                STDDEV_SAMP(pmts_dpd_303P) AS dpd303_std,
                approx_count_distinct(pmts_dpd_303P) AS dpd303_n_unique,
                MAX(closed_time_key) FILTER (WHERE closed_time_key IS NOT NULL) AS dpd303_recent_time_key,
                SUM(CASE WHEN pmts_overdue_1140A > 0 THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(COUNT(pmts_overdue_1140A), 0) AS overdue1140_overdue_rate,
                approx_count_distinct(pmts_overdue_1140A) AS overdue1140_n_unique,
                COUNT(pmts_overdue_1140A) AS overdue1140_non_null_count,
                SUM(CASE WHEN collater_valueofguarantee_1124L IS NULL THEN 1 ELSE 0 END) AS collater1124_null_count,
                approx_count_distinct(subjectroles_name_838M) AS subjectroles838_n_unique,
                MAX(active_time_key) FILTER (WHERE active_time_key IS NOT NULL)
                    - MIN(active_time_key) FILTER (WHERE active_time_key IS NOT NULL) AS active_date_duration,
                MIN(active_time_key) FILTER (WHERE active_time_key IS NOT NULL) AS active_date_min,
                MIN(closed_time_key) FILTER (WHERE closed_time_key IS NOT NULL) AS closed_date_min,
                ANY_VALUE(s.dpd303_longest_good_streak) AS dpd303_longest_good_streak
            FROM {a2_keyed} k
            LEFT JOIN {a2_streak_src} s USING (case_id, num_group1)
            GROUP BY k.case_id, k.num_group1
        ) TO '{l1_path}' (FORMAT PARQUET)
        """
    )
    a2_l1_src = f"read_parquet('{l1_path}')"

    # entropy(subjectroles_name_838M)需先算群內類別分布,再算熵
    con.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW a2_role_entropy AS
        WITH vc AS (
            SELECT case_id, num_group1, subjectroles_name_838M, COUNT(*) AS cnt
            FROM {a2_keyed}
            WHERE subjectroles_name_838M IS NOT NULL
            GROUP BY case_id, num_group1, subjectroles_name_838M
        ),
        withp AS (
            SELECT case_id, num_group1, cnt,
                cnt::DOUBLE / SUM(cnt) OVER (PARTITION BY case_id, num_group1) AS p
            FROM vc
        )
        SELECT case_id, num_group1, -SUM(p * LN(p)) AS subjectroles838_entropy
        FROM withp
        GROUP BY case_id, num_group1
        """
    )

    # --- L2:二次聚合 by case_id ---
    l2 = con.sql(
        f"""
        SELECT
            l1.case_id,
            MAX(dpd1073_trend) AS pmts_dpd_1073P_trend__max,
            AVG(dpd1073_trend) AS pmts_dpd_1073P_trend__mean,
            AVG(COALESCE(dpd1073_recent_time_key, 0)) AS pmts_dpd_1073P_recent_time_key__mean_fallback,
            AVG(dpd303_trend) AS pmts_dpd_303P_trend__mean,
            MAX(dpd303_longest_good_streak) AS pmts_dpd_303P_longest_good_streak__max,
            SUM(dpd303_sum_positive) / NULLIF(SUM(dpd303_positive_count), 0) AS pmts_dpd_303P_mean_positive__recomputed,
            SUM(dpd303_non_null_count)::BIGINT AS pmts_dpd_303P_non_null_count__sum,
            MAX(dpd303_std) AS pmts_dpd_303P_std__max,
            SUM(dpd303_n_unique)::BIGINT AS pmts_dpd_303P_n_unique__sum,
            AVG(COALESCE(dpd303_recent_time_key, 0)) AS pmts_dpd_303P_recent_time_key__mean_fallback,
            MAX(COALESCE(dpd303_recent_time_key, 0)) AS pmts_dpd_303P_recent_time_key__max_fallback,
            SUM(overdue1140_overdue_rate * overdue1140_non_null_count)
                / NULLIF(SUM(overdue1140_non_null_count), 0) AS pmts_overdue_1140A_overdue_rate__weighted_avg,
            SUM(overdue1140_n_unique)::BIGINT AS pmts_overdue_1140A_n_unique__sum,
            MAX(overdue1140_non_null_count) AS pmts_overdue_1140A_non_null_count__max,
            MAX(collater1124_null_count)::BIGINT AS collater_valueofguarantee_1124L_null_count__max,
            SUM(subjectroles838_n_unique)::BIGINT AS subjectroles_name_838M_n_unique__sum,
            AVG(re.subjectroles838_entropy) AS subjectroles_name_838M_entropy__mean,
            AVG(active_date_duration) AS pmts_year_1139T_pmts_month_158T_duration__mean,
            MIN(active_date_min) AS pmts_year_1139T_pmts_month_158T_min__min,
            MIN(closed_date_min) AS pmts_year_507T_pmts_month_706T_min__min
        FROM {a2_l1_src} l1
        LEFT JOIN a2_role_entropy re USING (case_id, num_group1)
        GROUP BY l1.case_id
        """
    ).pl()

    con.close()
    for p in (keyed_path, streak_path, l1_path):
        if os.path.exists(p):
            os.remove(p)

    # regr_slope 在樣本不足/x 無變異時回傳 NaN(而非 SQL NULL);統一轉為 null 避免下游誤判
    float_cols = [c for c, dt in l2.schema.items() if dt in (pl.Float32, pl.Float64)]
    l2 = l2.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
    return l2


# =============================================================================
# base + 組裝
# =============================================================================

FINAL_FEATURE_COLS = [
    "avgdpdtolclosure24_3658938P", "maxdbddpdtollast12m_3658940P", "pctinstlsallpaidlate1d_3546856L",
    "pmts_dpd_1073P_trend__max", "pmts_dpd_303P_longest_good_streak__max", "pctinstlsallpaidearl3d_427L",
    "maxdpdtolerance_577P_mean", "rejectreasonclient_4145042M_non_placeholder_ratio",
    "lastrejectreason_759M", "pmts_overdue_1140A_overdue_rate__weighted_avg",
    "pmts_dpd_1073P_trend__mean", "pmts_dpd_303P_trend__mean", "cntpmts24_3658933L",
    "disbursedcredamount_1113A", "maxdpdlast12m_727P", "maxdpdlast24m_143P", "numrejects9m_859L",
    "price_1097A", "education_1103M", "pmtaverage_3A", "requesttype_4525192L",
    "education_927M_appl", "incometype_1044T_appl", "numberofoutstandinstls_59L__min",
    "numberofoutstandinstls_59L__sum", "overdueamountmax_35A__mean", "overdueamountmax_35A__std",
    "pmts_dpd_303P_mean_positive__recomputed", "interestrate_311L", "maxdebt4_972A",
    "maxdpdtolerance_374P", "numinstlswithdpd10_728L", "numinstlswithdpd5_4187116L",
    "totalsettled_863A", "numberofoverdueinstlmax_1039L__min", "prolongationcount_599L__null_rate",
    "residualamount_856A__min", "totalamount_6A__sum", "totalamount_996A__mean",
    "pmts_dpd_303P_non_null_count__sum", "collater_valueofguarantee_1124L_null_count__max",
    "pmts_dpd_303P_recent_time_key__mean_fallback", "disbursementtype_67L", "maxdpdlast3m_392P",
    "numinstlswithoutdpd_562L", "numinstunpaidmaxest_4493212L", "days180_256L", "days30_165L",
    "days360_512L", "days90_310L", "amount_4527230A_sum_positive", "cancelreason_3545846M_mode",
    "education_1138M_mode", "familystate_447L_appl", "dpdmax_757P__null_rate",
    "monthlyinstlamount_674A__mean", "outstandingamount_362A__max", "overdueamountmax_155A__std",
    "maxannuity_4075009A", "empl_industry_691L_appl", "amount_4527230A_positive_count",
    "credtype_322L", "riskassesment_940T", "pmts_dpd_303P_n_unique__sum",
    "pmts_year_1139T_pmts_month_158T_duration__mean", "pmts_overdue_1140A_n_unique__sum",
    "pmts_dpd_1073P_recent_time_key__mean_fallback", "pmts_dpd_303P_recent_time_key__max_fallback",
    "opencred_647L", "subjectroles_name_838M_n_unique__sum", "subjectroles_name_838M_entropy__mean",
    "pmts_overdue_1140A_non_null_count__max", "pmts_dpd_303P_std__max",
    "registaddr_zipcode_184M_appl",
    # 3 個 D 欄改為 date_decision 差之 _days 版本(需求 4/5)
    "lastrejectdate_50D_days", "maxdpdinstldate_3546855D_days", "lastdelinqdate_224D_days",
    "pmts_year_507T_pmts_month_706T_min__min", "pmts_year_1139T_pmts_month_158T_min__min",
    # 7 個財務基礎欄(需求 6,聚合後保留為特徵)
    "totaldebt_9A", "maininc_215A", "mainoccupationinc_384A", "totaldebtoverduevalue_178A",
    "totaloutstanddebtvalue_39A", "amtdepositincoming_4809444A", "amtdebitoutgoing_4809440A",
    # 3 個財務比率特徵(需求 9)
    "dti_ratio", "overdue_debt_ratio", "deposit_to_debt_ratio",
    # 5 個衍生欄(需求 10)
    "reject_rate", "last_status", "tenure_years_max", "age_years_appl", "tenure_years_appl",  
]


def load_base(data_dir: str, split: str) -> pl.DataFrame:
    df = read_table(data_dir, split, "base").collect()
    return df.with_columns(pl.col("date_decision").str.strptime(pl.Date, strict=False))


def build_dataset(data_dir: str, split: str) -> pl.DataFrame:
    print(f"=== building {split} ===")
    base = load_base(data_dir, split)  # [資料工程] date_decision 已 parse 為 Date
    base_dates = base.select("case_id", "date_decision")

    print("  static...")
    static = build_static(data_dir, split, base_dates)
    print("  applprev_1...")
    applprev = build_applprev(data_dir, split)
    print("  tax_registry_a_1...")
    tax_a = build_tax_a(data_dir, split)
    print("  credit_bureau_a_1...")
    a1 = build_bureau_a1(data_dir, split)
    print("  credit_bureau_a_2 (duckdb, may take a while)...")
    a2 = build_bureau_a2(data_dir, split)
    print("  person_1...")
    person = build_person(data_dir, split, base_dates)
    print("  other_1...")
    other = build_other(data_dir, split)

    # [特徵工程] join 回 base(date_decision 本次保留於輸出)
    has_target = "target" in base.columns
    key_cols = ["case_id", "WEEK_NUM", "date_decision"] + (["target"] if has_target else [])
    out = base.select(key_cols)
    for part in [static, applprev, tax_a, a1, a2, person, other]:
        out = out.join(part, on="case_id", how="left")

    # [特徵工程] 財務比率(需 7 財務基礎欄皆已 join 齊全)
    out = build_financial_indicators(out)

    final_cols = key_cols + FINAL_FEATURE_COLS
    out = out.select(final_cols)
    print(f"  {split} shape: {out.shape}")
    return out


def main():
    df_train = build_dataset(TRAIN_DIR, "train")
    df_test = build_dataset(TEST_DIR, "test")

    # df_train.write_parquet(TRAIN_OUT)
    # df_test.write_parquet(TEST_OUT)
    # print(f"written: {TRAIN_OUT} {df_train.shape}")
    # print(f"written: {TEST_OUT} {df_test.shape}")
    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = main()



#訓練模型
train = df_train
test = df_test

remove = ("pmts_dpd_1073P_recent_time_key__mean_fallback",
"requesttype_4525192L",
"subjectroles_name_838M_entropy__mean",
"numinstunpaidmaxest_4493212L",
"pmtaverage_3A",
"empl_industry_691L_appl",
"registaddr_zipcode_184M_appl",
"pmts_year_1139T_pmts_month_158T_min__min",
"education_927M_appl",
"pmts_dpd_303P_recent_time_key__max_fallback",
"amount_4527230A_sum_positive",
"familystate_447L_appl",
"collater_valueofguarantee_1124L_null_count__max",
"totalamount_6A__sum",
"overdueamountmax_35A__mean",
"amount_4527230A_positive_count",
"overdueamountmax_35A__std",
"pmts_dpd_303P_recent_time_key__mean_fallback",
"pmts_dpd_303P_trend__mean",
"pmts_dpd_303P_non_null_count__sum",
"pmts_dpd_303P_mean_positive__recomputed",
"pmts_dpd_303P_std__max",
"pmts_dpd_303P_n_unique__sum",
"pmts_dpd_303P_longest_good_streak__max",
"pmts_year_507T_pmts_month_706T_min__min",
)

x_train = train.drop(["case_id","target","WEEK_NUM","date_decision"]+ list(remove))
y_train = train.select("target")
x_test = test.drop(["WEEK_NUM","date_decision"]+ list(remove))

# week_train_pd = (
#     train
#     .select("WEEK_NUM")
#     .to_numpy()
#     .ravel()
# )

# week_test_pd = (
#     test
#     .select("WEEK_NUM")
#     .to_numpy()
#     .ravel()
# )


x_train_pd = x_train.to_pandas()
y_train_pd = y_train.to_numpy().ravel()
x_test_pd = x_test.to_pandas()



categorical_cols = list(
    x_train_pd.select_dtypes(include=["object"]).columns
)

for col in categorical_cols:
    x_train_pd[col] = x_train_pd[col].astype("category")
    x_test_pd[col] = x_test_pd[col].astype("category")

    model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",

    n_estimators=500,

    learning_rate=0.05,
    reg_alpha=0.5,  # L1 正則化 (預設 0.0)
    reg_lambda=4,  # L2 正則化 (預設 0.0)
    max_depth=5,  # 限制樹深 (防止過擬合)
    num_leaves=31,

    random_state=42,

    n_jobs=-1
)

model.fit(
    x_train_pd,
    y_train_pd
)

## 這個就是輸出機率


x_test_pd = x_test_pd.set_index("case_id")

test_probability = pd.Series(model.predict_proba(x_test_pd)[:, 1], index=x_test_pd.index)

df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = test_probability
print("Check null: ", df_subm["score"].isnull().any())

df_subm.head()
print(df_subm)

df_subm.to_csv("submission.csv")