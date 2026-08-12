"""data/final_feature.csv 候選特徵的 VIF 多重共線性分析。

流程:
1. 讀取 data/final_feature.csv 的候選欄位清單,排除 date_decision(非特徵中繼欄)。
2. 以 DuckDB reservoir sampling 一次性取出固定列樣本(所有候選欄 + target 共用同一批列,
   以保留欄位間相關結構),而非逐欄各自取樣。
3. 對每欄依 dtype 分箱(重用 feature_selection_woe_iv 的決策樹/類別分箱),
   以 category_encoders.WOEEncoder 轉換為 WOE 數值欄,組成全數值 WOE 矩陣。
4. 以相關矩陣虛擬逆的對角線計算 VIF(與逐欄 OLS 迴歸法數學等價,且快得多);
   抽樣欄位另以 statsmodels.variance_inflation_factor 獨立驗證。
5. 分層(<5 低 / 5-10 中 / >10 高),輸出排名 CSV + 中文報告 MD(report-only,不自動剔除)。
"""

import argparse
import sys
import time

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import feature_selection_woe_iv as woe_mod

FEATURE_LIST_CSV = "data/final_feature.csv"
SRC = woe_mod.SRC
OUT_CSV = "data/final_feature_vif.csv"
OUT_MD = "data/final_feature_vif.md"

NON_FEATURE_COLS = {"date_decision"}
TARGET_COL = "target"
SAMPLE_N = 500_000
RANDOM_STATE = woe_mod.RANDOM_STATE
VIF_CAP = 1e6
N_STATSMODELS_CHECKS = 5


def load_candidate_columns(columns_file: str | None = None) -> tuple[list[str], dict[str, str]]:
    """回傳(去重排除 date_decision 後的欄位清單, {欄位: 來源} 對照)。

    預設讀取 `data/final_feature.csv` 的完整候選清單;若指定 columns_file
    (純文字檔,每行一個欄位名),則改分析該子集,但仍以 final_feature.csv 提供
    來源(source)對照(子集之外的清單不影響 VIF 計算,僅供顯示)。
    """
    df = pl.read_csv(FEATURE_LIST_CSV, encoding="utf8-lossy")
    name_col, source_col = df.columns[0], df.columns[1]
    df = df.unique(subset=[name_col])
    source_map = dict(zip(df[name_col].to_list(), df[source_col].to_list()))

    if columns_file:
        with open(columns_file, encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
        columns = sorted(set(raw) - NON_FEATURE_COLS)
        unknown = [c for c in columns if c not in source_map]
        if unknown:
            print(
                f"WARNING 下列欄位不在 final_feature.csv 來源清單中(source 將標記為空): {unknown}",
                file=sys.stderr,
            )
    else:
        columns = sorted(c for c in df[name_col].to_list() if c not in NON_FEATURE_COLS)
    return columns, source_map


def sample_data(columns: list[str]) -> pl.DataFrame:
    con = duckdb.connect()
    col_list = ", ".join(f'"{c}"' for c in columns)
    query = f"""
        SELECT {col_list}, "{TARGET_COL}"
        FROM '{SRC}'
        USING SAMPLE {SAMPLE_N} (reservoir, {RANDOM_STATE})
    """
    return con.sql(query).pl()


def classify(col: str, dtype: str) -> str:
    base = dtype.split("(")[0]
    if base in woe_mod.NUMERIC_TYPES:
        return "numeric"
    if base in woe_mod.BOOL_TYPES or base in woe_mod.STR_TYPES:
        return "categorical"
    if base in woe_mod.TS_TYPES:
        return "timestamp"
    raise ValueError(f"未知 dtype: {col} {dtype}")


def build_woe_column(
    series: pl.Series, var_type: str, target: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if var_type == "numeric":
        values = series.cast(pl.Float64, strict=False).to_numpy()
        labels, _ = woe_mod.tree_bin_labels(values, target, rng)
    elif var_type == "timestamp":
        values = series.cast(pl.Date, strict=False).cast(pl.Int32).cast(pl.Float64).to_numpy()
        labels, _ = woe_mod.tree_bin_labels(values, target, rng)
    else:  # categorical / boolean
        labels = woe_mod.cat_bin_labels(series)
    return woe_mod.woe_transform(labels, target)


def compute_vif_via_correlation(woe_matrix: np.ndarray) -> np.ndarray:
    """相關矩陣虛擬逆的對角線 = 標準化變數的 VIF(與逐欄 OLS 回歸法數學等價)。"""
    corr = np.corrcoef(woe_matrix, rowvar=False)
    inv = np.linalg.pinv(corr)
    vif = np.diag(inv).astype(float)
    return np.clip(vif, a_min=1.0, a_max=VIF_CAP)


def tier_of(vif: float) -> str:
    if vif < 5:
        return "低(可接受)"
    if vif <= 10:
        return "中(留意)"
    return "高(建議剔除)"


def verify_with_statsmodels(
    woe_matrix: np.ndarray, ok_columns: list[str], result: pl.DataFrame
) -> list[dict]:
    """挑選涵蓋不同 VIF 量級的欄位,以 statsmodels 逐欄 OLS 法獨立重算比對。"""
    n = len(ok_columns)
    k = min(N_STATSMODELS_CHECKS, n)
    idxs = sorted(set(np.linspace(0, n - 1, num=k, dtype=int).tolist()))
    exog = sm.add_constant(woe_matrix)

    checks = []
    print("--- statsmodels 交叉驗證(逐欄 OLS VIF,已加入常數項) ---")
    for rank_idx in idxs:
        name = result["column"][rank_idx]
        j = ok_columns.index(name)
        vif_sm = variance_inflation_factor(exog, j + 1)
        vif_corr = result.filter(pl.col("column") == name)["VIF"][0]
        checks.append({"column": name, "vif_corr": vif_corr, "vif_statsmodels": vif_sm})
        print(f"  {name}: corr-inverse={vif_corr:.4f}  statsmodels={vif_sm:.4f}")
    return checks


def write_markdown_report(
    result: pl.DataFrame, checks: list[dict], out_csv: str, out_md: str, subset_note: str = ""
):
    n_total = result.height
    tier_counts = result.group_by("tier").agg(pl.len().alias("n")).sort("n", descending=True)
    high = result.filter(pl.col("tier") == "高(建議剔除)")

    lines = []
    lines.append("# data/final_feature.csv 候選特徵 VIF 多重共線性報告\n")
    lines.append("## Context\n")
    lines.append(
        "團隊以多種方法(AUC+KS、Catboost、EBM、GBM、TOP150 入選-1~5、WOE)彙整出 "
        "`data/final_feature.csv` 195 個候選特徵。本報告排除非特徵中繼欄 `date_decision` "
        f"後,對其餘 {n_total} 個候選特徵計算 VIF (Variance Inflation Factor),檢查多重共線性,"
        "供後續人工決定去留(本報告為 report-only,不自動剔除欄位)。\n"
    )
    if subset_note:
        lines.append(f"{subset_note}\n")
    lines.append("## 方法\n")
    lines.append(
        f"1. 以 DuckDB reservoir sampling(固定 seed={RANDOM_STATE})從 `data/base_final.parquet` "
        f"取出 {SAMPLE_N:,} 列,所有候選欄共用同一批列以保留欄位間相關結構。\n"
    )
    lines.append(
        "2. 每欄以與 `feature_selection_woe_iv.py` 相同的決策樹分箱(數值/時間戳)或"
        "類別稀有併箱(類別欄)+ `category_encoders.WOEEncoder` 轉換為 WOE 數值欄,組成 "
        f"{SAMPLE_N:,} × {n_total} 的 WOE 矩陣。\n"
    )
    lines.append(
        "3. VIF 以相關矩陣虛擬逆(`numpy.linalg.pinv`)的對角線計算(與加入常數項的逐欄 "
        "OLS 迴歸法數學等價,但快得多);另抽樣欄位以 `statsmodels.variance_inflation_factor` "
        "獨立重算比對一致性(見下方驗證章節)。\n"
    )
    lines.append("## 分層門檻\n")
    lines.append("| VIF 區間 | 層級 |\n|---|---|\n")
    lines.append("| < 5 | 低(可接受) |\n")
    lines.append("| 5 – 10 | 中(留意) |\n")
    lines.append("| > 10 | 高(建議剔除) |\n")
    lines.append("\n## 結果摘要\n")
    lines.append(f"- 分析欄位總數:{n_total}\n")
    lines.append("\n| 層級 | 欄位數 |\n|---|---|\n")
    for row in tier_counts.iter_rows(named=True):
        lines.append(f"| {row['tier']} | {row['n']} |\n")

    lines.append("\n## 驗證:statsmodels 交叉比對\n")
    lines.append("| column | corr-inverse VIF | statsmodels VIF |\n|---|---|---|\n")
    for c in checks:
        lines.append(
            f"| {c['column']} | {c['vif_corr']:.4f} | {c['vif_statsmodels']:.4f} |\n"
        )

    lines.append("\n## 完整 VIF 排名\n")
    lines.append("| column | source | var_type | VIF | tier |\n")
    lines.append("|---|---|---|---|---|\n")
    for row in result.iter_rows(named=True):
        lines.append(
            f"| {row['column']} | {row['source']} | {row['var_type']} | "
            f"{row['VIF']:.4f} | {row['tier']} |\n"
        )

    if high.height > 0:
        lines.append("\n## 高 VIF 欄位(> 10,建議人工複查是否剔除)\n")
        lines.append("| column | source | VIF |\n|---|---|---|\n")
        for row in high.iter_rows(named=True):
            lines.append(f"| {row['column']} | {row['source']} | {row['VIF']:.4f} |\n")

    lines.append(f"\n完整結果見 `{out_csv}`。\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--columns-file",
        type=str,
        default=None,
        help="自訂欄位清單檔(純文字,每行一個欄位名)。預設讀取 final_feature.csv 全部候選欄。",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="輸出檔名前綴(寫到 data/<prefix>_vif.csv / data/<prefix>_vif.md)。"
        "預設沿用 final_feature_vif.csv / .md。",
    )
    args = parser.parse_args()

    out_csv = f"data/{args.out_prefix}_vif.csv" if args.out_prefix else OUT_CSV
    out_md = f"data/{args.out_prefix}_vif.md" if args.out_prefix else OUT_MD
    subset_note = (
        f"**本次為自訂子集重算**:從 194 個候選特徵中選取 {{n}} 個欄位重新計算 VIF"
        f"(欄位清單來源:`{args.columns_file}`)。"
        if args.columns_file
        else ""
    )

    t0 = time.time()
    columns, source_map = load_candidate_columns(args.columns_file)
    print(f"候選特徵數(已排除 date_decision): {len(columns)}")
    subset_note = subset_note.format(n=len(columns))

    schema_full = woe_mod.get_schema(SRC)
    var_types = {c: classify(c, schema_full[c]) for c in columns}

    print(f"取樣 {SAMPLE_N} 列...")
    df = sample_data(columns)
    print(f"樣本形狀: {df.shape}  elapsed={time.time() - t0:.0f}s")

    target = df[TARGET_COL].to_numpy().astype(np.int64)
    rng = np.random.default_rng(RANDOM_STATE)

    woe_cols: dict[str, np.ndarray] = {}
    n_total = len(columns)
    for i, col in enumerate(columns, start=1):
        try:
            woe_cols[col] = build_woe_column(df[col], var_types[col], target, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{n_total}] {col}: ERROR {exc}", file=sys.stderr)
        if i % 50 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] elapsed={time.time() - t0:.0f}s last={col}")

    ok_columns = [c for c in columns if c in woe_cols]
    if len(ok_columns) < len(columns):
        print(f"WARNING 略過 {len(columns) - len(ok_columns)} 個處理失敗的欄位", file=sys.stderr)

    woe_matrix = np.column_stack([woe_cols[c] for c in ok_columns])
    print(f"WOE 矩陣: {woe_matrix.shape}  elapsed={time.time() - t0:.0f}s")

    vif_values = compute_vif_via_correlation(woe_matrix)

    result = pl.DataFrame(
        {
            "column": ok_columns,
            "source": [source_map.get(c, "") for c in ok_columns],
            "var_type": [var_types[c] for c in ok_columns],
            "VIF": vif_values,
        }
    ).with_columns(
        pl.col("VIF").map_elements(tier_of, return_dtype=pl.Utf8).alias("tier")
    ).sort("VIF", descending=True)

    result.write_csv(out_csv)
    print(f"written: {out_csv}")

    checks = verify_with_statsmodels(woe_matrix, ok_columns, result)

    write_markdown_report(result, checks, out_csv, out_md, subset_note)
    print(f"written: {out_md}")

    tier_counts = result.group_by("tier").agg(pl.len().alias("n")).sort("n", descending=True)
    print(tier_counts)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
