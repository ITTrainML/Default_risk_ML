"""從 cat_cols.pkl 移除指定欄位並覆蓋原檔。

用法:
    python update_cat_cols.py
"""

import pickle
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAT_COLS_PATH = ROOT / "data" / "cat_cols.pkl"

# 要排除的欄位
DROP_COLS = ["mastercontrexist_109L"]


def main():
    with open(CAT_COLS_PATH, "rb") as f:
        cat_cols = pickle.load(f)

    print(f"原始欄位數: {len(cat_cols)}")

    removed = [c for c in DROP_COLS if c in cat_cols]
    missing = [c for c in DROP_COLS if c not in cat_cols]
    for c in missing:
        print(f"警告: {c} 不在 cat_cols 中，略過")

    if not removed:
        print("沒有欄位需要移除，原檔不變")
        return

    new_cat_cols = [c for c in cat_cols if c not in DROP_COLS]

    # 覆蓋前先備份
    backup_path = CAT_COLS_PATH.with_suffix(".pkl.bak")
    shutil.copy2(CAT_COLS_PATH, backup_path)
    print(f"已備份原檔至: {backup_path}")

    with open(CAT_COLS_PATH, "wb") as f:
        pickle.dump(new_cat_cols, f)

    print(f"已移除: {removed}")
    print(f"更新後欄位數: {len(new_cat_cols)}")
    print(f"已覆蓋: {CAT_COLS_PATH}")


if __name__ == "__main__":
    main()
