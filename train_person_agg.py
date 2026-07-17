"""
Step 3: Aggregate train_person_2_join_1.parquet by case_id
- Input: DATA/train_person_2_join_1.parquet (2.97M rows, 93 cols)
- Group by: case_id
- Output: DATA/train_person_agg.parquet

Aggregation strategies by column type:
  - Categorical/String/Bool: mode, mode_ratio, n_unique, entropy, null_count, null_rate, non_null_count
  - Numeric (Int64/Float64, non-key): mean, std, min, max, median, null_count
  - Date string (D suffix raw): earliest, latest, range_days, null_count
  - Aggregated metrics (_mode_ratio, _n_unique, _entropy, _non_null_count, _null_rate): mean, std, min, max
  - Aggregated metrics (_null_count): sum
  - Aggregated _mode (String): mode, n_unique
"""

import polars as pl
from datetime import date

INPUT_PATH = "DATA/train_person_2_join_1.parquet"
OUTPUT_PATH = "DATA/train_person_agg.parquet"
GROUP_KEY = "case_id"

TODAY = date.today()


def is_aggregated_metric(col: str) -> bool:
    """Check if column is already an aggregated metric from train_person_2."""
    suffixes = ["_mode", "_mode_ratio", "_n_unique", "_entropy",
                "_null_count", "_null_rate", "_non_null_count"]
    return any(col.endswith(s) for s in suffixes)


def is_date_col(col: str) -> bool:
    """Check if column is a raw date column (D suffix, not aggregated)."""
    return col.endswith("D") and not is_aggregated_metric(col)


def is_numeric_col(dtype: pl.DataType) -> bool:
    """Check if dtype is numeric."""
    return dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32, pl.UInt32, pl.UInt64]


def is_bool_col(dtype: pl.DataType) -> bool:
    return dtype == pl.Boolean


def is_string_col(dtype: pl.DataType) -> bool:
    return dtype == pl.String


def aggregate_categorical(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate categorical/string/bool columns: mode, mode_ratio, n_unique, entropy, null_count, null_rate, non_null_count."""
    prefix = col

    # Non-null subset for mode/entropy
    non_null = df.filter(pl.col(col).is_not_null())

    # Category counts per group
    freq = (
        non_null
        .group_by([group_key, col])
        .agg(pl.len().alias("cnt"))
    )

    # Group totals (non-null count)
    totals = (
        freq
        .group_by(group_key)
        .agg(pl.sum("cnt").alias("total"))
    )

    freq = (
        freq.join(totals, on=group_key)
        .with_columns((pl.col("cnt") / pl.col("total")).alias("p"))
    )

    # Mode & mode_ratio
    mode_df = (
        freq
        .sort([group_key, "cnt", col], descending=[False, True, False])
        .group_by(group_key, maintain_order=True)
        .first()
        .select([
            pl.col(group_key),
            pl.col(col).alias(f"{prefix}_mode"),
            (pl.col("cnt") / pl.col("total")).alias(f"{prefix}_mode_ratio"),
        ])
    )

    # Entropy
    entropy_df = (
        freq
        .with_columns((-pl.col("p") * pl.col("p").log(base=2)).alias("e_part"))
        .group_by(group_key)
        .agg(pl.sum("e_part").alias(f"{prefix}_entropy"))
    )

    # Basic stats
    stat_df = (
        df
        .group_by(group_key)
        .agg([
            pl.col(col).drop_nulls().n_unique().alias(f"{prefix}_n_unique"),
            pl.col(col).null_count().alias(f"{prefix}_null_count"),
            (pl.col(col).null_count() / pl.len()).alias(f"{prefix}_null_rate"),
            pl.col(col).count().alias(f"{prefix}_non_null_count"),
        ])
    )

    result = (
        stat_df
        .join(mode_df, on=group_key, how="left")
        .join(entropy_df, on=group_key, how="left")
    )
    return result


def aggregate_numeric(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate numeric columns: mean, std, min, max, median, null_count."""
    return (
        df
        .group_by(group_key)
        .agg([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).median().alias(f"{col}_median"),
            pl.col(col).null_count().alias(f"{col}_null_count"),
        ])
    )


def aggregate_date(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate date columns: earliest, latest, range_days, null_count."""
    parsed = df.with_columns(
        pl.col(col).str.to_date("%Y-%m-%d", strict=False).alias(f"_{col}_parsed")
    )
    stat_df = (
        parsed
        .group_by(group_key)
        .agg([
            pl.col(f"_{col}_parsed").min().alias(f"{col}_earliest"),
            pl.col(f"_{col}_parsed").max().alias(f"{col}_latest"),
            (
                pl.col(f"_{col}_parsed").max() - pl.col(f"_{col}_parsed").min()
            ).alias(f"{col}_range_days"),
            pl.col(col).null_count().alias(f"{col}_null_count"),
        ])
    )
    # Convert date to string for parquet compatibility
    stat_df = stat_df.with_columns([
        pl.col(f"{col}_earliest").dt.strftime("%Y-%m-%d"),
        pl.col(f"{col}_latest").dt.strftime("%Y-%m-%d"),
        pl.col(f"{col}_range_days").cast(pl.Int64),
    ])
    return stat_df


def aggregate_agg_metric(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate an already-aggregated metric (numeric) by case_id."""
    return (
        df
        .group_by(group_key)
        .agg([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).null_count().alias(f"{col}_null_count"),
        ])
    )


def aggregate_agg_mode(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate an already-aggregated mode (String) column."""
    return aggregate_categorical(df, col, group_key)


def aggregate_agg_null_count(df: pl.DataFrame, col: str, group_key: str) -> pl.DataFrame:
    """Aggregate null_count metric: sum across persons."""
    return (
        df
        .group_by(group_key)
        .agg([
            pl.col(col).sum().alias(f"{col}_sum"),
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).null_count().alias(f"{col}_null_count"),
        ])
    )


def main():
    print(f"Reading {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH)
    print(f"  Shape: {df.shape}, Columns: {df.width}")

    schema = df.schema
    all_cols = list(schema.keys())
    other_cols = [c for c in all_cols if c != GROUP_KEY]

    result = None

    for col in other_cols:
        dtype = schema[col]

        # Determine aggregation strategy
        if col == "num_group1":
            # Numeric but should be treated as count/n_unique per case
            agg_df = (
                df
                .group_by(GROUP_KEY)
                .agg([
                    pl.col(col).n_unique().alias("num_group1_n_unique"),
                    pl.col(col).count().alias("num_group1_count"),
                    pl.col(col).null_count().alias("num_group1_null_count"),
                ])
            )
        elif is_aggregated_metric(col):
            if col.endswith("_mode") and (is_string_col(dtype) or is_bool_col(dtype)):
                print(f"  Aggregating aggregated mode: {col}")
                agg_df = aggregate_agg_mode(df, col, GROUP_KEY)
            elif col.endswith("_null_count"):
                print(f"  Aggregating aggregated null_count: {col}")
                agg_df = aggregate_agg_null_count(df, col, GROUP_KEY)
            elif is_numeric_col(dtype):
                print(f"  Aggregating aggregated metric: {col}")
                agg_df = aggregate_agg_metric(df, col, GROUP_KEY)
            else:
                print(f"  Skipping aggregated column: {col} (unsupported type)")
                continue
        elif is_date_col(col) and is_string_col(dtype):
            print(f"  Aggregating date column: {col}")
            agg_df = aggregate_date(df, col, GROUP_KEY)
        elif is_bool_col(dtype):
            print(f"  Aggregating boolean column: {col}")
            agg_df = aggregate_categorical(df, col, GROUP_KEY)
        elif is_string_col(dtype):
            print(f"  Aggregating string column: {col}")
            agg_df = aggregate_categorical(df, col, GROUP_KEY)
        elif is_numeric_col(dtype):
            print(f"  Aggregating numeric column: {col}")
            agg_df = aggregate_numeric(df, col, GROUP_KEY)
        else:
            print(f"  Skipping column: {col} (dtype={dtype})")
            continue

        if result is None:
            result = agg_df
        else:
            result = result.join(agg_df, on=GROUP_KEY, how="left")

    print(f"\nFinal result shape: {result.shape}, Columns: {result.width}")
    print(f"Writing to {OUTPUT_PATH}...")
    result.write_parquet(OUTPUT_PATH, compression="zstd")
    print("Done!")


if __name__ == "__main__":
    main()