
#%%
#region[rgba(231,76,60,0.15)]

#### 先設置絕對路徑

import os

# 1. 定義你的目標資料夾路徑（前方加上 r 可以避免 Windows 反斜線引發的轉義字元錯誤）
target_path = r"C:\Users\ittraining\Desktop\git\Default_risk_ML\data"

# 2. 強制將 Python 的工作目錄切換到該資料夾
os.chdir(target_path)

# 3. 驗證目前的工作目錄（執行時可以順便確認有沒有切換成功）
print("【系統提示】目前的執行路徑已成功設定為：", os.getcwd())

# --- 接下來就可以直接用「相對路徑」讀取 DuckDB 或其他檔案了 ---
# 例如：
# import duckdb
# df = duckdb.query("SELECT * FROM 'your_file.csv' LIMIT 5").df()

#endregion
#%%
#region[rgba(241,196,60,0.15)]

##### 建立Codebook及檢查資料欄位是否相同

import pandas as pd
df_codebook = pd.read_csv("codebook.csv")

name_mapping = dict(zip(df_codebook["Variable"], df_codebook["Description_繁體中文"]))

import polars as pl

#【關鍵修復】處理你的 Codebook 字典，防止重複命名
# 假設你之前的對照表字典叫 rename_mapping，我們在這邊幫它「自動去重」
seen_names = {}
safe_rename_mapping = {}

for old_col, new_col in name_mapping.items():
    if new_col in seen_names:
        seen_names[new_col] += 1
        safe_rename_mapping[old_col] = f"{new_col}_{seen_names[new_col]}"
    else:
        seen_names[new_col] = 0
        safe_rename_mapping[old_col] = new_col


# ####### 檔案欄位檢查

import glob

files = sorted(glob.glob("train_tax_registry_*.parquet"))

# 拿第一個檔案當作基準基準 (Baseline)
base_file = files[0]
base_schema = pl.scan_parquet(base_file).collect_schema()
all_match = True

print(f"以 {base_file} 作為基準（共 {len(base_schema)} 欄），開始進行全面比對...\n")

for f in files[1:]:
    current_schema = pl.scan_parquet(f).collect_schema()
    
    if base_schema == current_schema:
        print(f"✅ {f} 欄位與基準完全一致！")
    else:
        all_match = False
        print(f"❌ {f} 發現不一致！")
        
        # 找出名字不同的欄位
        base_cols = set(base_schema.names())
        curr_cols = set(current_schema.names())
        
        missing_in_curr = base_cols - curr_cols
        extra_in_curr = curr_cols - base_cols
        
        if missing_in_curr:
            print(f"   缺少了基準有的欄位: {missing_in_curr}")
        if extra_in_curr:
            print(f"   多了基準沒有的欄位: {extra_in_curr}")

if all_match:
    print("\n🎉 檢查完成：所有資料集的欄位名稱與型態完全相同，可以安心直接合併！")

#endregion
#%%
#region[rgba(52,152,219,0.15)]

#### 批量讀取資料且合併，並用Codebook替換欄位名稱

lazy_df1 = pl.scan_parquet("train_static_0_*.parquet")

current_cols = set(lazy_df1.columns)
filtered_mapping = {k: v for k, v in safe_rename_mapping.items() if k in current_cols}

lazy_df1 = lazy_df1.rename(filtered_mapping)

print(lazy_df1.head(5).collect())

shape = (lazy_df1.select(pl.len()).collect().item(), len(lazy_df1.columns))
print(f"資料集形狀 (列, 欄)：{shape}")

lazy_df1.sink_parquet('train_static_0_merge.parquet')

#endregion
#%%
#region[rgba(46,204,113,0.15)]

#### 使用DUCKDB 資料庫讀取檔案，並建立線上UI

import duckdb
import polars as pl

con = duckdb.connect()

con.sql("""
    CREATE OR REPLACE VIEW bureau_a1_merge AS
    SELECT *
    FROM 'bureau_a1_merge.parquet'
""")
con.sql("CALL start_ui();")

#需斷點
print("hellow")
#endregion


# %%
#region[rgba(200,100,130,0.15)]

### 檢查資料集的遺失值狀況

import duckdb
import pandas as pd


PARQUET_PATH = r"train_tax_registry_c_1.parquet"
OUTPUT_CSV = "missing_report.csv"

con = duckdb.connect()

source_sql = f"read_parquet('{PARQUET_PATH}')"



# 總筆數
total_rows = con.execute(f"""
    SELECT COUNT(*) 
    FROM {source_sql}
""").fetchone()[0]

# 取得所有欄位名稱
schema_df = con.execute(f"""
    DESCRIBE SELECT * 
    FROM {source_sql}
""").df()

results = []

for col in schema_df["column_name"]:
    safe_col = f'"{col}"'

    missing_count = con.execute(f"""
        SELECT COUNT(*) - COUNT({safe_col})
        FROM {source_sql}
    """).fetchone()[0]

    missing_pct = missing_count / total_rows 

    results.append({
        "column_name": col,
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 4)
    })

missing_df = pd.DataFrame(results)

missing_df["column_name"] = missing_df["column_name"].map(safe_rename_mapping).fillna(missing_df["column_name"])

missing_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

missing_df

#endregion
# %%
#region[rgba(26,188,258,0.15)]

############# 逆向工程拆解bureau_a2_merge 

### Step 1：先用 DuckDB 建立輕量分析表 
## 不要讀全部欄位，只讀 key + 年月 + 逾期 + 類型欄位。

import duckdb

con = duckdb.connect("home_credit_reverse.duckdb")

A2_PATH = "bureau_a2_merge.parquet"

con.execute(f"""
CREATE OR REPLACE TABLE a2_probe AS
SELECT
    case_id,
    num_group1,
    num_group2,

    "存續合約的繳款年份（num_group1－現有合約，num_group2－繳款）。" AS active_year,
    "存續合約的繳款月份（num_group1－已終止合約，num_group2－繳款）。" AS active_month,

    "已結清授信合約的繳款年份（num_group1－已終止合約，num_group2－繳款）。" AS closed_year,
    "已結清合約的繳款月份（num_group1－現有合約，num_group2－繳款）。" AS closed_month,

    "存續合約該筆繳款的逾期天數（num_group1－現有合約，num_group2－繳款）。" AS active_dpd,
    "依聯徵中心，已終止合約該筆繳款的逾期天數（num_group1－已終止合約，num_group2－繳款）。" AS closed_dpd,

    "存續合約的逾期繳款（num_group1－現有合約，num_group2－繳款）。" AS active_overdue,
    "已結清合約的逾期繳款（num_group1－已終止合約，num_group2－繳款）。" AS closed_overdue

FROM read_parquet('{A2_PATH}')
""")

### Step 2：先看整體資料量與 key 結構

# con.execute("""
# SELECT
#     COUNT(*) AS rows,
#     COUNT(DISTINCT case_id) AS n_case,
#     COUNT(DISTINCT case_id || '-' || num_group1) AS n_case_group1,
#     COUNT(DISTINCT case_id || '-' || num_group1 || '-' || num_group2) AS n_case_group1_group2
# FROM a2_probe
# """).df()

### Step 3 ：確認num_group2 是不是付款序號

## 一筆 contract 平均有幾筆 depth2 history
history_summary = con.execute("""
SELECT
    AVG(cnt) AS avg_history,
    MIN(cnt) AS min_history,
    MAX(cnt) AS max_history,
    QUANTILE_CONT(cnt, 0.5) AS median_history
FROM (
    SELECT
        case_id,
        num_group1,
        COUNT(*) AS cnt
    FROM a2_probe
    GROUP BY case_id, num_group1
)
""").df()

history_summary

## 檢查 num_group2 是否像連續序號：
group2_check = con.execute("""
SELECT
    case_id,
    num_group1,
    MIN(num_group2) AS min_g2,
    MAX(num_group2) AS max_g2,
    COUNT(*) AS cnt,
    COUNT(DISTINCT num_group2) AS n_unique_g2,
    MAX(num_group2) - MIN(num_group2) + 1 AS expected_cnt,
    CASE
        WHEN COUNT(*) = MAX(num_group2) - MIN(num_group2) + 1
        THEN 1
        ELSE 0
    END AS is_continuous
FROM a2_probe
GROUP BY case_id, num_group1
LIMIT 50
""").df()

group2_check

## 看整體比例：

continuous_summary = con.execute("""
SELECT
    COUNT(*) AS n_contracts,
    SUM(is_continuous) AS continuous_contracts,
    AVG(is_continuous) AS continuous_ratio
FROM (
    SELECT
        case_id,
        num_group1,
        CASE
            WHEN COUNT(*) = MAX(num_group2) - MIN(num_group2) + 1
            THEN 1
            ELSE 0
        END AS is_continuous
    FROM a2_probe
    GROUP BY case_id, num_group1
)
""").df()

continuous_summary
## 重點看 continuous_ratio。如果接近 1
# num_group2 幾乎可以視為每筆合約下的連續歷史序號。

#endregion
# %%
