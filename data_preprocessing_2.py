import math

import polars as pl

SRC = "data/train_applprev_join.parquet"
OUT = "data/train_applprev_1_agg.parquet"

M_PLACEHOLDER = "a55475b1"
REFERENCE_YEAR = 2024

# applprev_2_agg 欄位:由 data_preprocessing.py 在 (case_id, num_group1) 層級
# 聚合產出的指標欄,本階段做二次聚合到 case_id 層級。
# 僅讀取實際會用到的指標欄(cacccard / credacc 的 ratio/entropy 無變異,不讀)。
A2_COLS = [
    "cacccardblochreas_147M_mode",
    "cacccardblochreas_147M_null_count",
    "conts_type_509L_mode",
    "conts_type_509L_mode_ratio",
    "conts_type_509L_n_unique",
    "conts_type_509L_entropy",
    "conts_type_509L_null_count",
    "credacc_cards_status_52L_mode",
    "credacc_cards_status_52L_null_count",
]

# a2 的 null_count 欄各僅 1 列 NULL(整表無紀錄),補 0
A2_NULLCOUNT_COLS = [
    "cacccardblochreas_147M_null_count",
    "conts_type_509L_null_count",
    "credacc_cards_status_52L_null_count",
]

KEEP_COLS = [
    "case_id",
    "creationdate_885D",
    "actualdpd_943P",
    "byoccupationinc_3656910L",
    "cancelreason_3545846M",
    "credacc_credlmt_575A",
    "currdebt_94A",
    "education_1138M",
    "employedfrom_700D",
    "mainoccupationinc_437A",
    "maxdpdtolerance_577P",
    "outstandingdebt_522A",
    "profession_152M",
    "rejectreason_755M",
    "rejectreasonclient_4145042M",
    "revolvingaccount_394A",
    "status_219L",
] + A2_COLS

# 需做 99% 分位數截尾(winsorize)的連續型欄位
# 不含 actualdpd_943P(p99=0,截尾會抹掉全部逾期訊號)
# 不含 revolvingaccount_394A(帳戶編號,非金額)
WINSORIZE_COLS = [
    "byoccupationinc_3656910L",
    "credacc_credlmt_575A",
    "currdebt_94A",
    "mainoccupationinc_437A",
    "maxdpdtolerance_577P",
    "outstandingdebt_522A",
]

# 大寫 A 結尾欄位:負值視為異常值,清除為 NULL
A_SUFFIX_COLS = [
    "credacc_credlmt_575A",
    "currdebt_94A",
    "mainoccupationinc_437A",
    "outstandingdebt_522A",
    "revolvingaccount_394A",
]

# 聚合後補 0 的欄位(結構性缺失:無合約/無帳戶即無此金額)
FILL_ZERO_COLS = [
    "actualdpd_943P",
    "maxdpdtolerance_577P",
    "currdebt_94A",
    "outstandingdebt_522A",
    "credacc_credlmt_575A",
]

# 5 個離散型 M 類別欄
CAT_COLS = [
    "cancelreason_3545846M",
    "education_1138M",
    "profession_152M",
    "rejectreason_755M",
    "rejectreasonclient_4145042M",
]


def build_cat_agg(
    df: pl.DataFrame, col: str, keys: list[str], placeholder: bool = True
) -> pl.DataFrame:
    """群組內類別欄的統計指標。placeholder=True 時額外輸出非佔位值比例
    (供 M 欄位使用);L 欄位無 'a55475b1' 佔位值,設為 False 略過死特徵。"""
    # drop_nulls:整組皆空的 case 不進 vc,join 後為 NULL(結構性缺失不補值)
    vc = df.drop_nulls(col).group_by(keys + [col]).agg(pl.len().alias("cnt"))
    exprs = [
        # 以次數排序取眾數;次數相同時再依類別值排序確保結果可重現
        pl.col(col)
        .sort_by(["cnt", col], descending=[True, False])
        .first()
        .alias(f"{col}_mode"),
        (pl.col("cnt").max() / pl.col("cnt").sum()).alias(f"{col}_mode_ratio"),
        pl.len().alias(f"{col}_n_unique"),
        pl.col("cnt").entropy(base=math.e, normalize=True).alias(f"{col}_entropy"),
    ]
    if placeholder:
        exprs.append(
            (
                pl.col("cnt").filter(pl.col(col) != M_PLACEHOLDER).sum()
                / pl.col("cnt").sum()
            )
            .fill_null(0.0)
            .alias(f"{col}_non_placeholder_ratio")
        )
    return vc.group_by(keys).agg(exprs)


def main():
    df = pl.read_parquet(SRC, columns=KEEP_COLS)

    df = df.with_columns(
        pl.col("creationdate_885D").str.strptime(pl.Date, strict=False),
        pl.col("employedfrom_700D").str.strptime(pl.Date, strict=False),
    )

    # 規則3:大寫 A 結尾欄位負值清除(實測無負值,作為防護)
    df = df.with_columns(
        [
            pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in A_SUFFIX_COLS
        ]
    )

    # 規則4:大寫 D 結尾欄位,就業起始日晚於申請日不合理 -> 設為 NULL
    df = df.with_columns(
        pl.when(pl.col("employedfrom_700D") > pl.col("creationdate_885D"))
        .then(None)
        .otherwise(pl.col("employedfrom_700D"))
        .alias("employedfrom_700D")
    )

    # 就業年資衍生欄:以 2024 年為基準
    df = df.with_columns(
        (
            REFERENCE_YEAR
            - (
                pl.col("employedfrom_700D").dt.year()
                + (pl.col("employedfrom_700D").dt.month() - 1) / 12
            )
        ).alias("tenure_years")
    )

    # 99%PR 截尾(winsorize,不刪列)
    p99 = df.select([pl.col(c).quantile(0.99).alias(c) for c in WINSORIZE_COLS]).row(0)
    df = df.with_columns(
        [
            pl.when(pl.col(c) > p99[i]).then(p99[i]).otherwise(pl.col(c)).alias(c)
            for i, c in enumerate(WINSORIZE_COLS)
        ]
    )

    # 缺失值處理:結構性缺失補 0
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in FILL_ZERO_COLS])

    # a2 的 null_count 欄補 0(各僅 1 列 NULL)
    df = df.with_columns([pl.col(c).fill_null(0) for c in A2_NULLCOUNT_COLS])

    # revolvingaccount_394A 為帳戶編號,轉二元旗標(null -> 0)
    df = df.with_columns(
        pl.col("revolvingaccount_394A").is_not_null().cast(pl.Int8).alias("has_revolving")
    )

    # 依 case_id、creationdate 排序,使 last() = 最近一次申請
    df = df.sort(["case_id", "creationdate_885D"])

    KEYS = ["case_id"]

    # 真實卡片凍結事件:非佔位值且非空(NULL 視同「無凍結」)
    real_card_block = (
        pl.col("cacccardblochreas_147M_mode").fill_null(M_PLACEHOLDER) != M_PLACEHOLDER
    )

    numeric_agg = df.group_by(KEYS).agg(
        # actualdpd_943P
        pl.col("actualdpd_943P").max().alias("actualdpd_943P_max"),
        pl.col("actualdpd_943P").mean().alias("actualdpd_943P_mean"),
        (pl.col("actualdpd_943P") > 0).sum().alias("actualdpd_943P_n_events"),
        # maxdpdtolerance_577P
        pl.col("maxdpdtolerance_577P").max().alias("maxdpdtolerance_577P_max"),
        pl.col("maxdpdtolerance_577P").mean().alias("maxdpdtolerance_577P_mean"),
        # currdebt_94A
        pl.col("currdebt_94A").sum().alias("currdebt_94A_sum"),
        pl.col("currdebt_94A").max().alias("currdebt_94A_max"),
        pl.col("currdebt_94A").last().alias("currdebt_94A_last"),
        # outstandingdebt_522A
        pl.col("outstandingdebt_522A").sum().alias("outstandingdebt_522A_sum"),
        pl.col("outstandingdebt_522A").max().alias("outstandingdebt_522A_max"),
        pl.col("outstandingdebt_522A").last().alias("outstandingdebt_522A_last"),
        # credacc_credlmt_575A
        pl.col("credacc_credlmt_575A").max().alias("credacc_credlmt_575A_max"),
        pl.col("credacc_credlmt_575A").last().alias("credacc_credlmt_575A_last"),
        # mainoccupationinc_437A
        pl.col("mainoccupationinc_437A").last().alias("mainoccupationinc_437A_last"),
        pl.col("mainoccupationinc_437A").max().alias("mainoccupationinc_437A_max"),
        pl.col("mainoccupationinc_437A").mean().alias("mainoccupationinc_437A_mean"),
        # byoccupationinc_3656910L
        pl.col("byoccupationinc_3656910L").last().alias("byoccupationinc_3656910L_last"),
        pl.col("byoccupationinc_3656910L").mean().alias("byoccupationinc_3656910L_mean"),
        pl.col("byoccupationinc_3656910L").null_count().truediv(pl.len()).alias(
            "byoccupationinc_3656910L_null_ratio"
        ),
        # tenure_years
        pl.col("tenure_years").sum().alias("tenure_years_sum"),
        pl.col("tenure_years").last().alias("tenure_years_last"),
        pl.col("tenure_years").max().alias("tenure_years_max"),
        # education_1138M 另需 last
        pl.col("education_1138M").last().alias("education_1138M_last"),
        # revolvingaccount_394A -> 二元旗標
        pl.col("has_revolving").max().alias("has_revolving_any"),
        pl.col("has_revolving").mean().alias("has_revolving_ratio"),
        # status_219L 拒絕率 / 取消率特徵
        pl.len().alias("n_prev_apps"),
        pl.col("status_219L").is_not_null().sum().alias("n_status_valid"),
        (pl.col("status_219L") == "D").sum().alias("n_rejected"),
        (
            (pl.col("status_219L") == "D").sum()
            / pl.col("status_219L").is_not_null().sum()
        ).alias("reject_rate"),
        (
            (pl.col("status_219L") == "T").sum()
            / pl.col("status_219L").is_not_null().sum()
        ).alias("cancel_rate"),
        pl.col("status_219L").last().alias("last_status"),
        # === applprev_2_agg 二次聚合 ===
        # conts_type_509L(聯絡方式,訊號最豐富,完整指標組)
        pl.col("conts_type_509L_mode").last().alias("conts_type_509L_mode_last"),
        pl.col("conts_type_509L_mode_ratio").mean().alias(
            "conts_type_509L_mode_ratio_mean"
        ),
        pl.col("conts_type_509L_mode_ratio").min().alias(
            "conts_type_509L_mode_ratio_min"
        ),
        pl.col("conts_type_509L_n_unique").max().alias("conts_type_509L_n_unique_max"),
        pl.col("conts_type_509L_n_unique").mean().alias("conts_type_509L_n_unique_mean"),
        pl.col("conts_type_509L_entropy").max().alias("conts_type_509L_entropy_max"),
        pl.col("conts_type_509L_entropy").mean().alias("conts_type_509L_entropy_mean"),
        pl.col("conts_type_509L_null_count").sum().alias(
            "conts_type_509L_null_count_sum"
        ),
        # cacccardblochreas_147M(卡片凍結,稀有事件旗標)
        real_card_block.any().cast(pl.Int8).alias("has_card_block_any"),
        real_card_block.sum().alias("n_card_block_apps"),
        pl.col("cacccardblochreas_147M_null_count").sum().alias(
            "cacccardblochreas_147M_null_count_sum"
        ),
        # credacc_cards_status_52L(先前信用帳戶卡片狀態,持卡與負面狀態旗標)
        pl.col("credacc_cards_status_52L_mode").is_not_null().any().cast(pl.Int8).alias(
            "has_card_any"
        ),
        pl.col("credacc_cards_status_52L_mode").is_not_null().sum().alias("n_card_apps"),
        pl.col("credacc_cards_status_52L_mode").drop_nulls().last().alias(
            "card_status_last"
        ),
        pl.col("credacc_cards_status_52L_mode")
        .is_in(["CANCELLED", "BLOCKED"])
        .fill_null(False)
        .any()
        .cast(pl.Int8)
        .alias("card_cancelled_or_blocked_any"),
        pl.col("credacc_cards_status_52L_null_count").sum().alias(
            "credacc_cards_status_52L_null_count_sum"
        ),
    )

    result = numeric_agg

    # 5 個 M 類別欄:mode / mode_ratio / n_unique / entropy / non_placeholder_ratio
    for col in CAT_COLS:
        stats = build_cat_agg(df, col, KEYS, placeholder=True)
        result = result.join(stats, on=KEYS, how="left")

    # conts_type_509L_mode 的「跨申請眾數」(mode of modes):長期慣用聯絡方式
    # 及其偏好穩定度;L 欄無佔位值 -> placeholder=False
    mode_of_modes = build_cat_agg(df, "conts_type_509L_mode", KEYS, placeholder=False)
    result = result.join(mode_of_modes, on=KEYS, how="left")

    result = result.sort(KEYS)
    result.write_parquet(OUT)
    print(result.shape)
    print(result.columns)


if __name__ == "__main__":
    main()
