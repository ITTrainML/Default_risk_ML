"""
Step 2: Join cleaned train_person_1 with train_person_2_aggregated
- Left table: train_person_1_cleaned.parquet (37 cols)
- Right table: train_person_2_aggregated.parquet (58 cols)
- Join key: [case_id, num_group1]
- Join type: LEFT JOIN
Output: DATA/train_person_2_join_1.parquet
"""

import polars as pl

LEFT_PATH = "DATA/train_person_1_cleaned.parquet"
RIGHT_PATH = "DATA/train_person_2_aggregated.parquet"
OUTPUT_PATH = "DATA/train_person_2_join_1.parquet"
JOIN_KEY = ["case_id", "num_group1"]


def main():
    print("Reading left table (train_person_1_cleaned)...")
    left = pl.read_parquet(LEFT_PATH)
    print(f"  Shape: {left.shape}, columns: {left.width}")

    print("Reading right table (train_person_2_aggregated)...")
    right = pl.read_parquet(RIGHT_PATH)
    print(f"  Shape: {right.shape}, columns: {right.width}")

    # Validate join keys exist
    for key in JOIN_KEY:
        if key not in left.columns:
            raise ValueError(f"Left table missing join key: {key}")
        if key not in right.columns:
            raise ValueError(f"Right table missing join key: {key}")

    print(f"\nJoining on {JOIN_KEY}...")
    result = left.join(right, on=JOIN_KEY, how="left")

    print(f"Result shape: {result.shape}, columns: {result.width}")

    # Check null counts from right-side columns after join
    right_cols = [c for c in right.columns if c not in JOIN_KEY]
    null_stats = result.select(
        [pl.col(c).null_count().alias(c) for c in right_cols[:5]]
    )
    print(f"Null counts in first 5 right-side columns:\n{null_stats}")

    print(f"\nWriting to {OUTPUT_PATH}...")
    result.write_parquet(OUTPUT_PATH, compression="zstd")
    print("Done!")


if __name__ == "__main__":
    main()