import math

import polars as pl

SRC = "data/train_person_2_join_1.parquet"
OUT = "data/train_person_agg.parquet"

KEYS = ["case_id"]

# 本人(num_group1=0)直取的類別/布林欄(relationshiptoclient_* 與 remitter_829L 不納入:
# 前者僅描述關係人、後者為死特徵)
PERSON_APPL_CAT_COLS = [
    "contaddr_district_15M",
    "education_927M",
    "empl_employedtotal_800L",
    "empl_industry_691L",
    "familystate_447L",
    "housetype_905L",
    "housingtype_772L",
    "incometype_1044T",
    "maritalst_703L",
    "persontype_1072L",
    "persontype_792L",
    "registaddr_district_1083M",
    "registaddr_zipcode_184M",
    "role_1084L",
    "safeguarantyflag_411L",
    "type_25L",
]

# 需 99%PR 截尾(winsorize,不刪列)的連續型欄位
WINSORIZE_COLS = ["childnum_185L", "mainoccupationinc_384A"]

# person_2_aggregated 的 8 個類別來源欄
P2_SOURCE_COLS = [
    "addres_district_368M",
    "addres_role_871L",
    "addres_zip_823M",
    "conts_role_79M",
    "empls_economicalst_849M",
    "empls_employedfrom_796D",
    "empls_employer_name_740M",
    "relatedpersons_role_762T",
]
P2_APPL_METRICS = ("mode", "mode_ratio", "n_unique", "entropy")

READ_COLS = (
    [
        "case_id",
        "num_group1",
        "date_decision",
        "birth_259D",
        "birthdate_87D",
        "empl_employedfrom_271D",
        "childnum_185L",
        "mainoccupationinc_384A",
        "relationshiptoclient_415T",
        "relationshiptoclient_642T",
    ]
    + PERSON_APPL_CAT_COLS
    + [
        f"{c}_{m}"
        for c in P2_SOURCE_COLS
        for m in ("mode", "mode_ratio", "n_unique", "entropy", "null_count", "non_null_count")
    ]
)


def build_cat_agg(
    df: pl.DataFrame, col: str, keys: list[str], metrics: tuple[str, ...] = ("mode", "n_unique")
) -> pl.DataFrame:
    """群組內類別欄統計指標;mode 以次數/類別值排序取得,次數相同時可重現。
    整組皆空的群組不進 vc,join 後為 NULL(結構性缺失不補值)。"""
    vc = df.drop_nulls(col).group_by(keys + [col]).agg(pl.len().alias("cnt"))
    exprs = []
    if "mode" in metrics:
        exprs.append(
            pl.col(col)
            .sort_by(["cnt", col], descending=[True, False])
            .first()
            .alias(f"{col}_mode")
        )
    if "mode_ratio" in metrics:
        exprs.append((pl.col("cnt").max() / pl.col("cnt").sum()).alias(f"{col}_mode_ratio"))
    if "n_unique" in metrics:
        exprs.append(pl.len().alias(f"{col}_n_unique"))
    if "entropy" in metrics:
        exprs.append(
            pl.col("cnt").entropy(base=math.e, normalize=True).alias(f"{col}_entropy")
        )
    return vc.group_by(keys).agg(exprs)


def main():
    df = pl.read_parquet(SRC, columns=READ_COLS)

    df = df.with_columns(
        pl.col("date_decision").str.strptime(pl.Date, strict=False),
        pl.col("birth_259D").str.strptime(pl.Date, strict=False),
        pl.col("birthdate_87D").str.strptime(pl.Date, strict=False),
        pl.col("empl_employedfrom_271D").str.strptime(pl.Date, strict=False),
    )

    # 規則3:A 欄負值 -> null(實測無負值,防護性實作)
    df = df.with_columns(
        pl.when(pl.col("mainoccupationinc_384A") < 0)
        .then(None)
        .otherwise(pl.col("mainoccupationinc_384A"))
        .alias("mainoccupationinc_384A")
    )

    # 規則4:D 欄未來日(相對該列 date_decision) -> null
    df = df.with_columns(
        [
            pl.when(pl.col(c) > pl.col("date_decision"))
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ["birth_259D", "birthdate_87D", "empl_employedfrom_271D"]
        ]
    )

    # 日期衍生:年齡(coalesce 兩個生日欄備援)、就業年資
    df = df.with_columns(pl.coalesce(["birth_259D", "birthdate_87D"]).alias("_birth"))
    df = df.with_columns(
        ((pl.col("date_decision") - pl.col("_birth")).dt.total_days() / 365.25).alias(
            "age_years"
        ),
        (
            (pl.col("date_decision") - pl.col("empl_employedfrom_271D")).dt.total_days()
            / 365.25
        ).alias("tenure_years"),
    )

    # 規則2:連續欄 99%PR 截尾(不刪列;p99 現場計算,不寫死)
    p99 = df.select([pl.col(c).quantile(0.99).alias(c) for c in WINSORIZE_COLS]).row(0)
    df = df.with_columns(
        [
            pl.when(pl.col(c) > p99[i]).then(p99[i]).otherwise(pl.col(c)).alias(c)
            for i, c in enumerate(WINSORIZE_COLS)
        ]
    )

    all_cases = df.select(KEYS).unique()

    # === A. 本人特徵(num_group1=0,直接取值,後綴 _appl) ===
    appl_cols = ["age_years", "tenure_years"] + WINSORIZE_COLS + PERSON_APPL_CAT_COLS
    applicant = (
        df.filter(pl.col("num_group1") == 0)
        .select(KEYS + appl_cols)
        .rename({c: f"{c}_appl" for c in appl_cols})
    )

    # === C-1. person_2 指標:本人列直取(後綴 _appl) ===
    p2_appl_cols = [f"{c}_{m}" for c in P2_SOURCE_COLS for m in P2_APPL_METRICS]
    p2_appl = (
        df.filter(pl.col("num_group1") == 0)
        .select(KEYS + p2_appl_cols)
        .rename({c: f"{c}_appl" for c in p2_appl_cols})
    )

    # === C-2. person_2 指標:全案彙總(含本人+關係人所有列) ===
    p2_case_agg = df.group_by(KEYS).agg(
        [pl.col(f"{c}_null_count").sum().alias(f"{c}_null_count_sum") for c in P2_SOURCE_COLS]
        + [
            pl.col(f"{c}_non_null_count").sum().alias(f"{c}_non_null_count_sum")
            for c in P2_SOURCE_COLS
        ]
        + [pl.col(f"{c}_n_unique").max().alias(f"{c}_n_unique_max") for c in P2_SOURCE_COLS]
        + [pl.col(f"{c}_entropy").max().alias(f"{c}_entropy_max") for c in P2_SOURCE_COLS]
    )
    p2_case_fill_cols = [f"{c}_n_unique_max" for c in P2_SOURCE_COLS] + [
        f"{c}_entropy_max" for c in P2_SOURCE_COLS
    ]

    # === B. 關係人特徵(num_group1>0) ===
    related = df.filter(pl.col("num_group1") > 0)
    n_related = related.group_by(KEYS).agg(pl.len().alias("n_related_persons"))
    rel_415 = build_cat_agg(
        related, "relationshiptoclient_415T", KEYS, metrics=("mode", "n_unique")
    )
    rel_642 = build_cat_agg(related, "relationshiptoclient_642T", KEYS, metrics=("mode",))
    persontype_related = build_cat_agg(related, "persontype_792L", KEYS, metrics=("mode",)).rename(
        {"persontype_792L_mode": "persontype_792L_mode_related"}
    )
    rel_addr = build_cat_agg(
        related, "registaddr_district_1083M", KEYS, metrics=("n_unique",)
    ).rename({"registaddr_district_1083M_n_unique": "rel_addr_n_unique"})

    result = (
        all_cases.join(applicant, on=KEYS, how="left")
        .join(p2_appl, on=KEYS, how="left")
        .join(p2_case_agg, on=KEYS, how="left")
        .join(n_related, on=KEYS, how="left")
        .join(rel_415, on=KEYS, how="left")
        .join(rel_642, on=KEYS, how="left")
        .join(persontype_related, on=KEYS, how="left")
        .join(rel_addr, on=KEYS, how="left")
    )

    # 結構性缺失補 0:無關係人的 case -> 0;無任何 p2 紀錄的 case -> n_unique/entropy 0
    result = result.with_columns(
        pl.col("n_related_persons").fill_null(0),
        *[pl.col(c).fill_null(0) for c in p2_case_fill_cols],
    )

    result = result.sort(KEYS)
    result.write_parquet(OUT)
    print(result.shape)
    print(result.columns)


if __name__ == "__main__":
    main()
