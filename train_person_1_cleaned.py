"""
Step 1: Clean train_person_1.parquet
- Rule 2: Cap continuous features at 99% percentile (skip if P99=0)
- Rule 3: For columns with 'A' suffix, replace negative values with null
- Rule 4: For date columns (D suffix), if date is in the future → set to null
Output: DATA/train_person_1_cleaned.parquet
"""

import polars as pl
from datetime import date

INPUT_PATH = "DATA/train_person_1.parquet"
OUTPUT_PATH = "DATA/train_person_1_cleaned.parquet"

TODAY = date.today()

# Columns to process (25 specified columns)
CONTINUOUS_COLS = ["mainoccupationinc_384A", "childnum_185L"]  # Rule 2 & 3
NEGATIVE_TO_NULL_COLS = ["mainoccupationinc_384A"]  # Rule 3 (A suffix)
DATE_COLS = ["birth_259D", "birthdate_87D", "empl_employedfrom_271D"]  # Rule 4


def cap_at_p99(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Cap values above 99th percentile at the 99th percentile value."""
    p99 = df.select(pl.col(col).quantile(0.99)).item()
    if p99 is None or p99 <= 0:
        print(f"  {col}: P99={p99}, skip capping")
        return df
    before = df.filter(pl.col(col) > p99).height
    df = df.with_columns(
        pl.when(pl.col(col) > p99)
        .then(pl.lit(p99))
        .otherwise(pl.col(col))
        .alias(col)
    )
    print(f"  {col}: P99={p99:.2f}, capped {before} values")
    return df


def negative_to_null(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Replace negative values with null."""
    before = df.filter(pl.col(col) < 0).height
    df = df.with_columns(
        pl.when(pl.col(col) < 0)
        .then(pl.lit(None))
        .otherwise(pl.col(col))
        .alias(col)
    )
    print(f"  {col}: replaced {before} negative values with null")
    return df


def clean_future_dates(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Set future dates to null."""
    before = df.filter(pl.col(col).is_not_null()).height
    df = df.with_columns(
        pl.col(col).alias("_date_str")
    ).with_columns(
        pl.when(
            pl.col("_date_str").str.to_date("%Y-%m-%d", strict=False) > TODAY
        )
        .then(pl.lit(None))
        .otherwise(pl.col("_date_str"))
        .alias(col)
    ).drop("_date_str")
    after = df.filter(pl.col(col).is_not_null()).height
    removed = before - after
    if removed > 0:
        print(f"  {col}: removed {removed} future dates")
    return df


def main():
    print("Reading train_person_1.parquet...")
    df = pl.read_parquet(INPUT_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    # Rule 2: Cap continuous features at P99
    print("\n[Rule 2] Capping continuous features at P99...")
    for col in CONTINUOUS_COLS:
        if col in df.columns:
            df = cap_at_p99(df, col)

    # Rule 3: Negative values to null for A-suffix columns
    print("\n[Rule 3] Replacing negative values with null...")
    for col in NEGATIVE_TO_NULL_COLS:
        if col in df.columns:
            df = negative_to_null(df, col)

    # Rule 4: Clean future dates
    print("\n[Rule 4] Cleaning future dates...")
    for col in DATE_COLS:
        if col in df.columns:
            df = clean_future_dates(df, col)

    # Save
    print(f"\nWriting to {OUTPUT_PATH}...")
    df.write_parquet(OUTPUT_PATH, compression="zstd")
    print(f"Done! Shape: {df.shape}")


if __name__ == "__main__":
    main()