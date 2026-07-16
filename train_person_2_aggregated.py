#Aggregate train_person_2 by case_id and num_group1

import polars as pl


INPUT_PATH = "data/train_person_2.parquet"
OUTPUT_PATH = "data/train_person_2_aggreagated.parquet"
GROUP_COLUMNS = ["case_id", "num_group1"]
FEATURES = [
    "addres_district_368M",
    "addres_role_871L",
    "addres_zip_823M",
    "conts_role_79M",
    "empls_economicalst_849M",
    "empls_employedfrom_796D",
    "empls_employer_name_740M",
    "relatedpersons_role_762T",
]


def aggregate_categorical_feature(source: pl.LazyFrame, feature: str) -> pl.DataFrame:
    """Return seven categorical aggregation metrics for one feature."""
    feature_source = source.select([*GROUP_COLUMNS, feature])
    category_counts = (
        feature_source.filter(pl.col(feature).is_not_null())
        .group_by([*GROUP_COLUMNS, feature])
        .len()
        .rename({"len": "category_count"})
        .collect()
    )
    grouped = (
        feature_source.group_by(GROUP_COLUMNS)
        .agg(
            [
                pl.col(feature).null_count().alias(f"{feature}_null_count"),
                (pl.col(feature).null_count() / pl.len()).alias(f"{feature}_null_rate"),
                pl.col(feature).count().alias(f"{feature}_non_null_count"),
                pl.col(feature).drop_nulls().n_unique().alias(f"{feature}_n_unique"),
            ]
        )
        .collect()
    )

    mode_stats = (
        category_counts.sort(
            [*GROUP_COLUMNS, "category_count", feature],
            descending=[False, False, True, False],
        )
        .group_by(GROUP_COLUMNS, maintain_order=True)
        .first()
        .rename({feature: f"{feature}_mode", "category_count": "mode_count"})
    )
    entropy_stats = category_counts.group_by(GROUP_COLUMNS).agg(
        (
            -(
                (pl.col("category_count") / pl.col("category_count").sum())
                * (pl.col("category_count") / pl.col("category_count").sum()).log(2)
            ).sum()
        ).alias(f"{feature}_entropy")
    )

    return (
        grouped.join(mode_stats, on=GROUP_COLUMNS, how="left")
        .join(entropy_stats, on=GROUP_COLUMNS, how="left")
        .with_columns(
            pl.when(pl.col(f"{feature}_non_null_count") > 0)
            .then(pl.col("mode_count") / pl.col(f"{feature}_non_null_count"))
            .otherwise(None)
            .alias(f"{feature}_mode_ratio"),
            pl.col(f"{feature}_entropy").fill_null(0.0),
        )
        .drop("mode_count")
        .select(
            [
                *GROUP_COLUMNS,
                f"{feature}_mode",
                f"{feature}_mode_ratio",
                f"{feature}_n_unique",
                f"{feature}_entropy",
                f"{feature}_null_count",
                f"{feature}_null_rate",
                f"{feature}_non_null_count",
            ]
        )
    )


def main() -> None:
    source = pl.scan_parquet(INPUT_PATH)
    schema = source.collect_schema()
    required_columns = [*GROUP_COLUMNS, *FEATURES]
    missing_columns = [column for column in required_columns if column not in schema]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    aggregated = aggregate_categorical_feature(source, FEATURES[0])
    for feature in FEATURES[1:]:
        aggregated = aggregated.join(
            aggregate_categorical_feature(source, feature),
            on=GROUP_COLUMNS,
            how="inner",
            validate="1:1",
        )

    aggregated.sort(GROUP_COLUMNS).write_parquet(OUTPUT_PATH)
    print(f"Wrote {aggregated.height} rows and {aggregated.width} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
