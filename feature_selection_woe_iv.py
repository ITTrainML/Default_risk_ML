"""base_final.parquet 單變量 WOE/IV 特徵篩選。

流程(每個自變數):
1. 依 dtype 分箱 -> 數值/時間戳欄用決策樹(target-aware)分箱,類別欄直接當箱(稀有類別併箱)。
2. 用 category_encoders.WOEEncoder 對分箱後的離散欄轉換,取得每箱 WOE。
3. 用每箱的 good/bad 分布(套用與 WOEEncoder 相同的 regularization)自算 IV。
4. 依 IV 分層並套用 keep/drop 門檻(IV >= 0.02 保留),輸出排名 CSV + 中文報告 MD。
"""

import argparse
import sys
import time
from datetime import date, timedelta

import category_encoders as ce
import duckdb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier

SRC = "data/base_final.parquet"
OUT_CSV = "data/base_final_woe_iv.csv"
OUT_MD = "data/base_final_woe_iv.md"

EXCLUDE_COLS = {"case_id", "date_decision", "MONTH", "WEEK_NUM", "target"}
TARGET_COL = "target"

MISSING_LABEL = "__MISSING__"
RARE_LABEL = "__RARE__"
CAT_RARE_MIN_COUNT = 1000

REGULARIZATION = 0.5
MAX_LEAF_NODES = 8
MIN_SAMPLES_LEAF_FRAC = 0.02
MIN_SAMPLES_LEAF_ABS = 1000
TREE_SAMPLE_CAP = 300_000
RANDOM_STATE = 42

NUMERIC_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    "DOUBLE", "FLOAT", "HUGEINT", "DECIMAL",
}
BOOL_TYPES = {"BOOLEAN"}
STR_TYPES = {"VARCHAR"}
TS_TYPES = {"TIMESTAMP", "DATE"}


def get_schema(path: str) -> dict[str, str]:
    con = duckdb.connect()
    desc = con.sql(f"DESCRIBE SELECT * FROM '{path}'").pl()
    return dict(zip(desc["column_name"], desc["column_type"]))


def classify_columns(schema: dict[str, str]):
    numeric_cols, cat_cols, ts_cols, other = [], [], [], []
    for col, dtype in schema.items():
        if col in EXCLUDE_COLS:
            continue
        base = dtype.split("(")[0]
        if base in NUMERIC_TYPES:
            numeric_cols.append(col)
        elif base in BOOL_TYPES:
            cat_cols.append(col)
        elif base in STR_TYPES:
            cat_cols.append(col)
        elif base in TS_TYPES:
            ts_cols.append(col)
        else:
            other.append(col)
    return numeric_cols, cat_cols, ts_cols, other


def fit_tree_thresholds(
    x_valid: np.ndarray, y_valid: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """對(抽樣後的)非空值擬合單特徵決策樹,回傳排序後的唯一切點(分界值)陣列。
    由於樹只用單一特徵,所有內部節點的 threshold 排序後即為葉節點對應的區間邊界,
    等價於 tree.apply() 的分箱結果,但直接給出可讀的區間邊界。"""
    n_valid = x_valid.shape[0]
    if n_valid < 50 or len(np.unique(y_valid)) < 2 or np.unique(x_valid).shape[0] < 2:
        return np.array([])

    if n_valid > TREE_SAMPLE_CAP:
        idx = rng.choice(n_valid, size=TREE_SAMPLE_CAP, replace=False)
        x_fit, y_fit = x_valid[idx], y_valid[idx]
    else:
        x_fit, y_fit = x_valid, y_valid

    min_leaf = max(MIN_SAMPLES_LEAF_ABS, int(MIN_SAMPLES_LEAF_FRAC * x_fit.shape[0]))
    tree = DecisionTreeClassifier(
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(x_fit.reshape(-1, 1), y_fit)

    t = tree.tree_
    is_leaf = t.children_left == -1
    return np.unique(t.threshold[~is_leaf])


def apply_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """依切點將值分箱為 bin_0..bin_k 字串(right=True,即 <=threshold 歸入較低箱,與
    sklearn 樹的 <=threshold 走左子節點慣例一致)。"""
    if thresholds.size == 0:
        return np.full(values.shape[0], "bin_0", dtype=object)
    idx = np.digitize(values, thresholds, right=True)
    return np.array([f"bin_{i}" for i in idx], dtype=object)


def tree_bin_labels(values: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """數值欄(含由時間戳轉換而來的天數序數)決策樹分箱,回傳(箱標籤陣列, 切點陣列)。"""
    n = values.shape[0]
    mask = ~np.isnan(values)
    labels = np.full(n, MISSING_LABEL, dtype=object)
    if mask.sum() == 0:
        return labels, np.array([])

    x_valid = values[mask]
    y_valid = target[mask]
    thresholds = fit_tree_thresholds(x_valid, y_valid, rng)
    labels[mask] = apply_thresholds(x_valid, thresholds)
    return labels, thresholds


def format_numeric_intervals(
    thresholds: np.ndarray, has_missing: bool, is_timestamp: bool = False
) -> str:
    """把切點陣列轉成人類可讀的區間字串,如 'bin_0:(-inf, 12.50] | bin_1:(12.50, 45.00] | ...'。"""

    def fmt(v: float) -> str:
        if is_timestamp:
            return (date(1970, 1, 1) + timedelta(days=int(round(v)))).isoformat()
        return f"{v:.4g}"

    if thresholds.size == 0:
        parts = ["(-inf, inf)"]
    else:
        edges = [fmt(t) for t in thresholds]
        parts = [f"(-inf, {edges[0]}]"]
        parts += [f"({edges[i]}, {edges[i + 1]}]" for i in range(len(edges) - 1)]
        parts.append(f"({edges[-1]}, inf)")

    labeled = [f"bin_{i}:{d}" for i, d in enumerate(parts)]
    if has_missing:
        labeled.append(MISSING_LABEL)
    return " | ".join(labeled)


def cat_bin_labels(series: pl.Series) -> np.ndarray:
    """類別/布林欄:null -> __MISSING__,稀有類別(< CAT_RARE_MIN_COUNT)併入 __RARE__。"""
    s = series.cast(pl.Utf8).fill_null(MISSING_LABEL)
    counts = s.value_counts()
    rare = set(
        counts.filter(
            (pl.col("count") < CAT_RARE_MIN_COUNT) & (pl.col(s.name) != MISSING_LABEL)
        )[s.name].to_list()
    )
    if rare:
        s = s.map_elements(lambda v: RARE_LABEL if v in rare else v, return_dtype=pl.Utf8)
    return s.to_numpy()


def woe_transform(labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    """對分箱標籤呼叫 category_encoders.WOEEncoder,回傳與 labels 等長的逐列 WOE 值。"""
    df_pd = pd.DataFrame({"bin": labels})
    enc = ce.WOEEncoder(cols=["bin"], regularization=REGULARIZATION)
    return enc.fit_transform(df_pd, target)["bin"].to_numpy()


def woe_and_iv(labels: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """對分箱標籤取得各箱 WOE(經 category_encoders),並自算 IV(同一 regularization)。"""
    woe_values = woe_transform(labels, target)

    tmp = pl.DataFrame({"bin": labels, "target": target, "woe": woe_values})
    agg = tmp.group_by("bin").agg(
        pl.col("target").sum().alias("bad"),
        pl.len().alias("n"),
        pl.col("woe").first().alias("woe"),
    )
    n_bins = agg.height

    total_bad = int(target.sum())
    total_good = int(target.shape[0] - total_bad)
    if total_bad == 0 or total_good == 0:
        return 0.0, n_bins

    agg = agg.with_columns((pl.col("n") - pl.col("bad")).alias("good"))
    reg = REGULARIZATION
    agg = agg.with_columns(
        ((pl.col("bad") + reg) / (total_bad + 2 * reg)).alias("dist_bad"),
        ((pl.col("good") + reg) / (total_good + 2 * reg)).alias("dist_good"),
    )
    iv = float(((agg["dist_bad"] - agg["dist_good"]) * agg["woe"]).sum())
    return iv, n_bins


def tier_of(iv: float) -> str:
    if iv < 0.02:
        return "無預測力"
    if iv < 0.10:
        return "弱"
    if iv < 0.30:
        return "中"
    if iv < 0.50:
        return "強"
    return "可疑(疑洩漏/過強)"


def decision_of(iv: float) -> str:
    return "keep" if iv >= 0.02 else "drop"


def process_column(name: str, dtype: str, var_type: str, target: np.ndarray, rng):
    bin_intervals = ""
    if var_type == "numeric":
        s = pl.read_parquet(SRC, columns=[name])[name]
        values = s.cast(pl.Float64, strict=False).to_numpy()
        missing_rate = float(np.isnan(values).mean())
        labels, thresholds = tree_bin_labels(values, target, rng)
        bin_intervals = format_numeric_intervals(thresholds, missing_rate > 0, is_timestamp=False)
    elif var_type == "timestamp":
        s = pl.read_parquet(SRC, columns=[name])[name]
        values = s.cast(pl.Date, strict=False).cast(pl.Int32).cast(pl.Float64).to_numpy()
        missing_rate = float(np.isnan(values).mean())
        labels, thresholds = tree_bin_labels(values, target, rng)
        bin_intervals = format_numeric_intervals(thresholds, missing_rate > 0, is_timestamp=True)
    else:  # categorical / boolean
        s = pl.read_parquet(SRC, columns=[name])[name]
        missing_rate = float(s.is_null().mean())
        labels = cat_bin_labels(s)

    iv, n_bins = woe_and_iv(labels, target)
    return {
        "column": name,
        "dtype": dtype,
        "var_type": var_type,
        "n_bins": n_bins,
        "missing_rate": round(missing_rate, 6),
        "IV": round(iv, 6),
        "tier": tier_of(iv),
        "decision": decision_of(iv),
        "bin_intervals": bin_intervals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 個特徵欄(測試用)")
    args = parser.parse_args()

    t0 = time.time()
    schema = get_schema(SRC)
    numeric_cols, cat_cols, ts_cols, other = classify_columns(schema)
    print(
        f"numeric={len(numeric_cols)} categorical={len(cat_cols)} "
        f"timestamp={len(ts_cols)} other={len(other)}"
    )
    if other:
        print("WARNING 未分類欄位(將略過):", other)

    all_cols = (
        [(c, "numeric") for c in numeric_cols]
        + [(c, "categorical") for c in cat_cols]
        + [(c, "timestamp") for c in ts_cols]
    )
    if args.limit:
        all_cols = all_cols[: args.limit]

    target = pl.read_parquet(SRC, columns=[TARGET_COL])[TARGET_COL].to_numpy().astype(np.int64)
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    n_total = len(all_cols)
    for i, (col, var_type) in enumerate(all_cols, start=1):
        try:
            row = process_column(col, schema[col], var_type, target, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{n_total}] {col}: ERROR {exc}", file=sys.stderr)
            row = {
                "column": col,
                "dtype": schema[col],
                "var_type": var_type,
                "n_bins": 0,
                "missing_rate": None,
                "IV": None,
                "tier": "error",
                "decision": "drop",
                "bin_intervals": "",
            }
        rows.append(row)
        if i % 50 == 0 or i == n_total:
            elapsed = time.time() - t0
            print(f"  [{i}/{n_total}] elapsed={elapsed:.0f}s last={col} IV={row.get('IV')}")

    result = pl.DataFrame(rows).sort("IV", descending=True, nulls_last=True)
    result.write_csv(OUT_CSV)

    kept = result.filter(pl.col("decision") == "keep").height
    dropped = result.filter(pl.col("decision") == "drop").height
    print(f"done in {time.time() - t0:.0f}s. total={result.height} keep={kept} drop={dropped}")
    print(f"written: {OUT_CSV}")

    write_markdown_report(result)
    print(f"written: {OUT_MD}")


def write_markdown_report(result: pl.DataFrame):
    n_total = result.height
    kept = result.filter(pl.col("decision") == "keep")
    dropped_n = result.filter(pl.col("decision") == "drop").height
    tier_counts = (
        result.group_by("tier").agg(pl.len().alias("n")).sort("n", descending=True)
    )

    lines = []
    lines.append("# base_final.parquet WOE/IV 單變量特徵篩選報告\n")
    lines.append("## Context\n")
    lines.append(
        "對 `data/base_final.parquet`(`data/base.parquet` LEFT JOIN "
        "`data/train_person_agg.parquet`,以 `case_id` 為鍵)中除 `case_id`、"
        "`date_decision`、`MONTH`、`WEEK_NUM`、`target` 以外的每個自變數,"
        "計算其對 `target`(1=違約、0=正常還款)的 Weight of Evidence (WOE) 與 "
        "Information Value (IV),以 IV 作為特徵去留依據。\n"
    )
    lines.append("## 方法\n")
    lines.append(
        "1. **分箱**:數值/時間戳欄以決策樹(`max_leaf_nodes=8`、"
        "`min_samples_leaf>=1000`、`class_weight=balanced`,對非空值擬合、"
        "抽樣上限 30 萬列,以控制執行時間)分箱;缺失值獨立成 `__MISSING__` 箱。"
        "類別欄以原值為箱,出現次數 < 1000 的稀有類別併入 `__RARE__`。\n"
    )
    lines.append(
        "2. **WOE**:對分箱後的離散欄以 `category_encoders.WOEEncoder`"
        "(`regularization=0.5`)轉換,取得每箱 WOE。\n"
    )
    lines.append(
        "3. **IV**:以各箱 good/bad 分布(套用與 WOEEncoder 相同的 regularization)"
        "自算 `IV = Σ(dist_bad - dist_good) × WOE`。\n"
    )
    lines.append("## 篩選門檻(Siddiqi 慣例)\n")
    lines.append("| IV 區間 | 層級 | decision |\n|---|---|---|\n")
    lines.append("| < 0.02 | 無預測力 | drop |\n")
    lines.append("| 0.02 – 0.10 | 弱 | keep |\n")
    lines.append("| 0.10 – 0.30 | 中 | keep |\n")
    lines.append("| 0.30 – 0.50 | 強 | keep |\n")
    lines.append("| > 0.50 | 可疑(疑洩漏/過強,建議人工複查) | keep |\n")
    lines.append("\n## 結果摘要\n")
    lines.append(f"- 受篩欄位總數:{n_total}\n")
    lines.append(f"- keep:{kept.height};drop:{dropped_n}\n")
    lines.append("\n| 層級 | 欄位數 |\n|---|---|\n")
    for row in tier_counts.iter_rows(named=True):
        lines.append(f"| {row['tier']} | {row['n']} |\n")

    lines.append("\n## Top 40 IV 排名\n")
    lines.append(
        "| column | var_type | n_bins | missing_rate | IV | tier | decision | bin_intervals |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")
    top = result.head(40)
    for row in top.iter_rows(named=True):
        lines.append(
            f"| {row['column']} | {row['var_type']} | {row['n_bins']} | "
            f"{row['missing_rate']} | {row['IV']} | {row['tier']} | {row['decision']} | "
            f"{row['bin_intervals']} |\n"
        )

    suspicious = result.filter(pl.col("tier").str.starts_with("可疑"))
    if suspicious.height > 0:
        lines.append("\n## 可疑欄位(IV > 0.5,建議人工複查是否洩漏)\n")
        lines.append("| column | var_type | n_bins | IV |\n|---|---|---|---|\n")
        for row in suspicious.iter_rows(named=True):
            lines.append(f"| {row['column']} | {row['var_type']} | {row['n_bins']} | {row['IV']} |\n")

    lines.append(f"\n完整結果見 `{OUT_CSV}`。\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
