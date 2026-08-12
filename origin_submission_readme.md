# origin_submission.ipynb 程式流程概要

> 本文件整理 `origin_submission.ipynb` 的完整程式流程，分為 **資料工程 (Data Engineering)**、**特徵工程 (Feature Engineering)** 與 **機器學習 (Machine Learning)** 三大部分。
>
> 本檔遵循 `build_model_dataset_plan.md` 的設計，以「資料工程 + 特徵工程」建構 train/test 建模資料集，再以單一 LightGBM 訓練並輸出 `submission.csv`。

## 競賽背景

- 競賽：Kaggle **Home Credit – Credit Risk Model Stability**（信用違約風險模型穩定性）。
- 任務：預測申請人是否違約（二元分類，`target` ∈ {0, 1}）。
- 資料架構：以 `base` 表（每個 `case_id` 一筆）為核心，橫向 join 多張 `depth_0 / depth_1 / depth_2` 的靜態與歷史交易表。
- 本檔同時建構 **train** 與 **test** 兩份資料集（`build_dataset` 對兩者皆執行）。

## 總覽流程圖

```
[1] 匯入套件(duckdb / polars / lightgbm / sklearn…) + 路徑與常量
        │
[2] 【資料工程】read_table / raw_glob：union 分區檔(vertical_relaxed)
        │   → transform_by_label：依欄位大寫尾碼標準化型別(D→Date, A/P→Float, M→Utf8, T/L→原生)
        │   → add_date_diff_days：date_decision − D 欄 → {col}_days
        ▼
[3] 【特徵工程】清洗：winsorize_p99 / clean_by_label / clean_future_d_cols
        │   → 依深度層級聚合：
        │       depth-0 static   → 直接 1:1 join（無聚合）
        │       depth-1 b1/tax/person/other → 一次聚合(stat_expr / 類別 mode / 衍生欄)
        │       depth-2 bureau_a_2 → DuckDB 兩次聚合(L1→case_id, L2→case_id)
        │   → 財務比率特徵 + 最終欄位 select(final_cols)
        ▼
[4] 輸出 df_train / df_test（build_dataset 對 train 與 test 各跑一次）
        │
[5] 【機器學習】drop 鍵/目標/remove 欄 → to_pandas → object 欄轉 category
        │   → LGBMClassifier(fit 全量 train，無 CV)
        │   → predict_proba(x_test)
        ▼
[6] 產生 submission.csv（以 sample_submission 的 case_id 對齊 score）
```

---

## 一、資料工程 (Data Engineering)

### 1.1 路徑與常量

- `ROOT = Path("/kaggle/input/competitions/home-credit-credit-risk-model-stability")`
- `TRAIN_DIR = ROOT / "parquet_files" / "train"`、`TEST_DIR = ROOT / "parquet_files" / "test"`
- `M_PLACEHOLDER = "a55475b1"`（類別空值遮罩值）
- `REFERENCE_YEAR = 2024`（衍生自由年數用參考年）
- `INVALID_YEAR_LO / HI = 2025, 2028`（`bureau_a2_merge_GroupRule.txt` 的無效年份範圍）

### 1.2 讀取與 union 分區檔

- `read_table(data_dir, split, table)`：若有 `{split}_{table}.parquet` 單檔則 `scan_parquet`；否則以 glob `{split}_{table}_*.parquet` 找到分區檔後 `pl.concat(..., how="vertical_relaxed")` 合併，回傳 LazyFrame。
- `raw_glob(...)`：回傳 DuckDB `read_parquet` 用的 glob pattern（單檔或多分區）。

### 1.3 標籤型別標準化 `transform_by_label`

依欄位名稱的**大寫尾碼**判斷型別（Polars）：

| 尾碼 | 型別 | 說明 |
|------|------|------|
| `D` | `Date` | 日期欄（`strptime`） |
| `P` | `Float64` | DPD / 逾期天數 → 數值 |
| `A` | `Float64` | 金額 → 數值 |
| `M` | `Utf8` | 遮罩類別 → 字串 |
| `T`, `L` | 維持原生 | 型別/tag 類，不強制轉 |

### 1.4 日期轉天數差 `add_date_diff_days`

- 對指定 D 欄：`{col}_days = (date_decision − {col}).dt.total_days()`。
- 原始 D 欄在特徵工程後移除，僅保留 `_days` 版本。

### 1.5 DuckDB 設定（處理超大表）

- `con.sql("SET memory_limit='8GB'")`、`SET preserve_insertion_order=false`、`SET threads TO max(1, cpu_count()//2)`。
- 適用於 depth-2 `credit_bureau_a_2`（**1.88 億列**），視窗/聚合函式（如 `regr_slope`）對超大表較穩健。

---

## 二、特徵工程 (Feature Engineering)

### 2.0 清洗函式（套用於各表）

- `winsorize_p99(df, cols)`：連續型依 **99% 分位數截尾**（`min(c, p99)`）；`p99 == 0`（或近似為 0）時不截尾，不刪列。
- `clean_by_label(df, cols)`：`A` 尾碼欄負值 → `null`；`_days` 欄負值 → `null`。
- `clean_future_d_cols(df, d_cols, ref_col)`：D 欄日期晚於參考日（`date_decision`）→ `null`（未來日期錯誤資料）。

### 2.1 聚合統計函式庫 `NUMERIC_STAT_FUNCS`

| 統計 | 說明 |
|------|------|
| `mean / max / min / median / std / sum` | 基本統計量 |
| `null_rate / null_count / non_null_count` | 缺失統計 |
| `n_unique` | 唯一值數 |
| `positive_count` | 正值計數 |
| `overdue_rate` | 逾時率 = 正值數 / 非 null 數 |
| `mean_positive / sum_positive` | 正值平均 / 正值總和 |

- `stat_expr(raw, stat)`：輸出欄名為 `{raw}__{stat}`（通用特徵）。
- `financial_stat_expr(raw, stat)`：財務基礎欄聚合，輸出**沿用原始欄名**（代表 case 層級值）。
- `cat_group_stats(df, keys, col, stats)`：類別欄分組統計 → `mode / n_unique / entropy / non_placeholder_ratio`。

### 2.2 depth-0：static（直接 join，無聚合）

- `STATIC_0_COLS`（約 30 欄，含財務基礎欄 `totaldebt_9A`、`maininc_215A`）。
- `STATIC_CB_0_COLS`：`education_1103M`、`pmtaverage_3A`、`riskassesment_940T`、`days30/90/180/360` 等。
- `build_static`：`transform_by_label` → 3 個 D 欄轉 `_days`（`lastrejectdate_50D_days` 等）→ 清洗+截尾 → drop 原 D 欄 → join `static_cb_0`。

### 2.3 depth-1：各表一次聚合（by case_id）

- **`build_bureau_a1`（credit_bureau_a_1）**：`A1_SPEC` 13 組「欄×統計」（`numberofoutstandinstls_59L__min/sum`、`overdueamountmax_35A__mean/std` 等）+ 財務欄 `totaldebtoverduevalue_178A/totaloutstanddebtvalue_39A` sum。
- **`build_applprev`（applprev_1）**：`maxdpdtolerance_577P_mean`、`status_219L` 拒絕數/有效數 → **`reject_rate`**、`last_status`、**`tenure_years_max`**（`REFERENCE_YEAR − employedfrom`）、類別 mode（`cancelreason_3545846M_mode`、`education_1138M_mode`）、`rejectreasonclient..._non_placeholder_ratio`。
- **`build_tax_a`（tax_registry_a_1）**：`amount_4527230A_sum_positive`、`amount_4527230A_positive_count`。
- **`build_person`（person_1）**：`birth_259D/birthdate_87D` coalesce → **`age_years_appl`**；`empl_employedfrom_271D` → **`tenure_years_appl`**；本人列（`num_group1==0`）直取 `*_appl` 類別欄（`education_927M_appl`、`incometype_1044T_appl` 等）；財務欄 `mainoccupationinc_384A` case 層級 `max`。
- **`build_other`（other_1）**：財務欄 `amtdepositincoming_4809444A`、`amtdebitoutgoing_4809440A` 各 `sum`。

---

### 2.4 depth-2：credit_bureau_a_2（DuckDB 兩次聚合）

資料量 **1.88 億列**，拆成多階段 checkpoint（`.tmp/a2_*.parquet`）避免記憶體暴增：

1. **p99 截尾**：`approx_quantile(col, 0.99)` 求近似分位數，0 則不截。
2. **清洗 + 時間鍵**：負值→null、`active_time_key`（`pmts_year_1139T*12+pmts_month_158T`）、`closed_time_key`（`pmts_year_507T*12+pmts_month_706T`），無效年份（2025–2028）與月份越界→null；`COPY ... TO keyed.parquet` 單趟寫出。
3. **gaps-and-islands**：`pmts_dpd_303P == 0` 的最長連續良善段 `dpd303_longest_good_streak`（`ROW_NUMBER` 分組法）。
4. **L1 聚合（by case_id, num_group1）**：`regr_slope(pmts_dpd_1073P, active_time_key)` 趨勢、`dpd303_sum_positive/positive_count/non_null_count/std/n_unique`、`overdue1140_overdue_rate`、`collater1124_null_count`、`active_date_duration/min/max` 等 → checkpoint。
5. **角色熵**：`subjectroles_name_838M` 群內分布 → `subjectroles838_entropy`。
6. **L2 聚合（by case_id）**：L1 結果二次聚合 → `pmts_dpd_1073P_trend__max/mean`、`pmts_dpd_303P_*`、`pmts_overdue_1140A_*`、`subjectroles_name_838M_*` 等最終特徵。
7. **NaN 清理**：`regr_slope` 資料不足時回傳 NaN → `fill_nan(None)` 統一轉 null；`COALESCE` 幫 recent_time_key 等填 0（`__mean_fallback`）。

### 2.5 財務比率特徵 `build_financial_indicators`

以 7 個財務基礎欄（`totaldebt_9A`、`maininc_215A`、`mainoccupationinc_384A`、`totaldebtoverduevalue_178A`、`totaloutstanddebtvalue_39A`、`amtdepositincoming_4809444A`、`amtdebitoutgoing_4809440A`）建構 3 個比率：

| 比率 | 公式 | 備註 |
|------|------|------|
| `dti_ratio` | `totaldebt_9A / income` | income = `coalesce(maininc_215A, mainoccupationinc_384A)` |
| `overdue_debt_ratio` | `totaldebtoverduevalue_178A / totaloutstanddebtvalue_39A` | 逾期 / 未償 |
| `deposit_to_debt_ratio` | `amtdepositincoming_4809444A / amtdebitoutgoing_4809440A` | 存入 / 支出 |

- `_safe_ratio`：分母為 null 或 0 時回傳 null（除零保護）。

### 2.6 組裝 `build_dataset`（train 與 test 共用）

1. `load_base`：讀 base + `date_decision` parse 成 Date。
2. 依序建構 7 個 part：`static / applprev / tax_a / a1 / a2 / person / other`。
3. 以 `case_id` **left join** 回 base（保留 `case_id / WEEK_NUM / date_decision / target(若有)`）。
4. `build_financial_indicators` 補 3 個財務比率。
5. `select(FINAL_FEATURE_COLS)`（約 80 個最終特徵，含 3 個 `_days` 欄、7 財務基礎欄、3 比率、5 衍生欄）。

---

## 三、機器學習 (Machine Learning)

### 3.1 資料準備

```python
remove = ("pmts_dpd_1073P_recent_time_key__mean_fallback", "requesttype_4525192L", ...)  # 排除欄

x_train = train.drop(["case_id", "target", "WEEK_NUM", "date_decision"] + list(remove))
y_train = train.select("target")
x_test  = test.drop(["WEEK_NUM", "date_decision"] + list(remove))   # 保留 case_id 供對齊
```

- `remove` 元組：顯式排除被視為不佳/可能洩漏的特徵（如 `pmts_dpd_303P_*` 部分、`education_927M_appl`、`familystate_447L_appl`、`pmtaverage_3A` 等）。

### 3.2 pandas 轉換與類別欄

- `x_train_pd = x_train.to_pandas()`、`y_train_pd = y_train.to_numpy().ravel()`、`x_test_pd = x_test.to_pandas()`。
- `categorical_cols = x_train_pd.select_dtypes(include=["object"]).columns`（M 尾碼類別欄）。
- 對每個 object 欄：`x_train_pd[col].astype("category")`、`x_test_pd[col].astype("category")`，供 LightGBM 原生類別處理。

### 3.3 模型與訓練（單一 LightGBM，無 CV）

```python
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)
model.fit(x_train_pd, y_train_pd)   # 全量 train 訓練
```

- 直接以完整 train 資料擬合，**不使用交叉驗證或驗證集**。

### 3.4 預測與提交

```python
x_test_pd = x_test_pd.set_index("case_id")
test_probability = pd.Series(model.predict_proba(x_test_pd)[:, 1], index=x_test_pd.index)

df_subm = pd.read_csv(ROOT / "sample_submission.csv").set_index("case_id")
df_subm["score"] = test_probability
print("Check null: ", df_subm["score"].isnull().any())
df_subm.to_csv("submission.csv")
```

- `predict_proba` 第 1 欄 = 違約機率 `score`。
- 以 `sample_submission.csv` 的 `case_id` 對齊輸出，檢查無 null 後寫入 `submission.csv`。

---

## 與 submission.ipynb 的主要差異

| 面向 | origin_submission.ipynb | submission.ipynb |
|------|-------------------------|------------------|
| 資料量處理 | DuckDB 兩階段聚合處理 1.88 億列 depth-2 表 | Polars 聚合 + 記憶體降型 |
| 型別規則 | 依大寫尾碼（D/A/P/M/T/L） | 依後綴（P/A/M/D） |
| 特徵 | 聚合統計 + 財務比率 + 5 衍生欄 + 角色熵/趨勢/連段 | 五組聚合(max/last/mean) + 年度校正 |
| 模型 | 單一 LGBM（無 CV、無集成） | CatBoost(GPU)+LGBM 各 5 折 + VotingModel 集成 |
| 提交 | 直接輸出機率 | 依 max_pmts_year 年度趨勢校正 |

---

## 附錄：主要函式對照表

| 函式 / 常數 | 階段 | 用途 |
|------------|------|------|
| `read_table` / `raw_glob` | 資料工程 | 讀取 / union 分區檔、DuckDB glob |
| `transform_by_label` | 資料工程 | 依大寫尾碼標準化型別 |
| `add_date_diff_days` | 資料工程 | 日期轉天數差 `_days` |
| `winsorize_p99` / `clean_by_label` / `clean_future_d_cols` | 特徵工程 | 截尾 / 負值清洗 / 未來日期清洗 |
| `NUMERIC_STAT_FUNCS` / `stat_expr` / `financial_stat_expr` | 特徵工程 | 聚合統計函式庫 |
| `cat_group_stats` | 特徵工程 | 類別欄 mode/entropy 等統計 |
| `build_static` / `build_bureau_a1` / `build_applprev` / `build_tax_a` / `build_person` / `build_other` | 特徵工程 | depth-0/1 各表特徵 |
| `build_bureau_a2` | 特徵工程 | depth-2 大表 DuckDB 兩階段聚合 |
| `build_financial_indicators` / `_safe_ratio` | 特徵工程 | 3 個財務比率 |
| `build_dataset` / `main` | 組裝 | 產出 df_train / df_test |
| `LGBMClassifier` / `model.fit` | 機器學習 | 單一 LightGBM 訓練 |
| `predict_proba` / `to_csv` | 機器學習 | 輸出 submission.csv |