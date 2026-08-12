#Joined train_person_2_aggregated and train_person_1 by case_id

#region Create case-level features from train_person_1 and train_person_2_aggregated.
"""Create case-level features from train_person_1 and train_person_2_aggregated."""

from math import log
from pathlib import Path
import polars as pl


DATA_DIR = Path("DATA")
PERSON_1_FILE = DATA_DIR / "train_person_1.parquet"
PERSON_2_FILE = DATA_DIR / "train_person_2_aggregated.parquet"
OUTPUT_FILE = DATA_DIR / "train_person_joined.parquet"
CASE_KEY = "case_id"
PERSON_1_EXCLUDED_COLUMNS = {CASE_KEY, "num_group1"}


def first_layer_features(source: pl.LazyFrame, feature: str) -> pl.LazyFrame:
    """Create seven categorical statistics for one train_person_1 column."""
    base = source.group_by(CASE_KEY).agg(
        pl.col(feature).null_count().alias(f"{feature}_null_count"),
        (pl.col(feature).null_count() / pl.len()).alias(f"{feature}_null_rate"),
        pl.col(feature).count().alias(f"{feature}_non_null_count"),
        pl.col(feature).drop_nulls().n_unique().alias(f"{feature}_n_unique"),
    )

    value_counts = (
        source.select([CASE_KEY, feature])
        .filter(pl.col(feature).is_not_null())
        .group_by([CASE_KEY, feature])
        .len(name="value_count")
    )
    probability = pl.col("value_count") / pl.col("value_count").sum()
    mode_and_entropy = (
        value_counts.sort(
            [CASE_KEY, "value_count", feature],
            descending=[False, True, False],
        )
        .group_by(CASE_KEY, maintain_order=True)
        .agg(
            pl.first(feature).alias(f"{feature}_mode"),
            (pl.first("value_count") / pl.col("value_count").sum()).alias(
                f"{feature}_mode_ratio"
            ),
            (-(probability * probability.log() / log(2))).sum().alias(
                f"{feature}_entropy"
            ),
        )
    )

    return base.join(mode_and_entropy, on=CASE_KEY, how="left").with_columns(
        pl.col(f"{feature}_entropy").fill_null(0.0),
        pl.col(f"{feature}_mode_ratio").fill_null(0.0),
    )


def aggregate_person_1() -> pl.LazyFrame:
    """Aggregate every non-key train_person_1 field to one row per case."""
    source = pl.scan_parquet(PERSON_1_FILE)
    features = [
        column
        for column in source.collect_schema().names()
        if column not in PERSON_1_EXCLUDED_COLUMNS
    ]

    result = first_layer_features(source, features[0])
    for feature in features[1:]:
        result = result.join(first_layer_features(source, feature), on=CASE_KEY, how="left")
    return result


def aggregate_person_2() -> pl.LazyFrame:
    """Apply the requested second-stage aggregation to each existing metric."""
    source = pl.scan_parquet(PERSON_2_FILE)
    metrics = [column for column in source.collect_schema().names() if column not in {CASE_KEY, "num_group1"}]
    expressions: list[pl.Expr] = []

    for metric in metrics:
        if metric.endswith("_mode"):
            expressions.extend(
                [
                    pl.col(metric).drop_nulls().mode().first().alias(f"{metric}_mode"),
                    pl.col(metric).drop_nulls().n_unique().alias(f"{metric}_n_unique"),
                ]
            )
        elif metric.endswith("_mode_ratio"):
            expressions.extend(
                [pl.col(metric).mean().alias(f"{metric}_mean"), pl.col(metric).min().alias(f"{metric}_min")]
            )
        elif metric.endswith("_n_unique") or metric.endswith("_entropy"):
            expressions.extend(
                [pl.col(metric).mean().alias(f"{metric}_mean"), pl.col(metric).max().alias(f"{metric}_max")]
            )
        elif metric.endswith("_null_count") or metric.endswith("_non_null_count"):
            expressions.append(pl.col(metric).sum().alias(f"{metric}_sum"))
        elif metric.endswith("_null_rate"):
            expressions.append(pl.col(metric).max().alias(f"{metric}_max"))
        else:
            raise ValueError(f"No second-stage rule configured for {metric}")

    return source.group_by(CASE_KEY).agg(expressions)


def main() -> None:
    person_1 = aggregate_person_1()
    person_2 = aggregate_person_2()
    joined = person_1.join(person_2, on=CASE_KEY, how="left")

    column_count = len(joined.collect_schema().names())
    if column_count != 334:
        raise ValueError(f"Expected 334 columns, found {column_count}")

    joined.sink_parquet(OUTPUT_FILE, compression="zstd")
    print(f"Saved {column_count} columns to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
#endregion
