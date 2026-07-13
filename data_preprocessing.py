import math

import polars as pl


SRC = "data/train_applprev_2.parquet"
OUT = "data/train_applprev_2_agg.parquet"
KEYS = ["case_id", "num_group1"]
CAT_COLS = [
    "cacccardblochreas_147M",
    "conts_type_509L",
    "credacc_cards_status_52L",
]


def main():
    df = pl.read_parquet(SRC, columns=KEYS + CAT_COLS)

    # 所有群組作為左表骨架（含整組皆空值的群組）
    result = df.select(KEYS).unique()

    for col in CAT_COLS:
        # 各群組該欄空值筆數（用原始列）
        nulls = df.group_by(KEYS).agg(
            pl.col(col).null_count().alias(f"{col}_null_count")
        )

        # 非空值的群組內 value counts，再彙整成統計指標
        vc = (
            df.drop_nulls(col)
              .group_by(KEYS + [col])
              .agg(pl.len().alias("cnt"))
        )
        stats = vc.group_by(KEYS).agg(
            # 以次數排序取眾數；次數相同時再依類別值排序確保結果可重現
            pl.col(col).sort_by(["cnt", col], descending=[True, False])
                       .first().alias(f"{col}_mode"),
            (pl.col("cnt").max() / pl.col("cnt").sum()).alias(f"{col}_mode_ratio"),
            pl.len().alias(f"{col}_n_unique"),
            # normalize=True 先把 cnt 轉成機率 p，base=e → -Σ p·ln(p)
            pl.col("cnt").entropy(base=math.e, normalize=True).alias(f"{col}_entropy"),
        )

        result = (result.join(stats, on=KEYS, how="left")
                        .join(nulls, on=KEYS, how="left"))

    result = result.sort(KEYS)
    result.write_parquet(OUT)
    print(result.shape)
    print(result.columns)


if __name__ == "__main__":
    main()
