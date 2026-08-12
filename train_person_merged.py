#Merge train_person_2 into train_person_1

"""Merge person-level source data with its aggregated features."""

from pathlib import Path

import polars as pl


DATA_DIR = Path("DATA")
LEFT_FILE = DATA_DIR / "train_person_1.parquet"
RIGHT_FILE = DATA_DIR / "train_person_2_aggregated.parquet"
OUTPUT_FILE = DATA_DIR / "train_person_merged.parquet"
JOIN_KEYS = ["case_id", "num_group1"]


def merge_person_data() -> None:
    """Left-join aggregated features onto every train_person_1 record."""
    merged = (
        pl.scan_parquet(LEFT_FILE)
        .join(pl.scan_parquet(RIGHT_FILE), on=JOIN_KEYS, how="left")
        .collect()
    )

    merged.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f"Saved {merged.height:,} rows and {merged.width} columns to {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_person_data()
