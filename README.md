#  Home Credit - Credit Risk Model Stability
專題題目為kaggle競賽題目，先將資料集下載放到專案
1. 請先至網址裡下載Dataset的parquet資料夾內檔案(可直接下載整份資料)
2. https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data
3. 將所有parquet檔放置專案中的data資料夾內

**data資料夾請不要進版控**

---

# submission.ipynb 程式流程概要

> 以下整理 `submission.ipynb` 的完整程式流程，分為 **資料工程 (Data Engineering)**、**特徵工程 (Feature Engineering)** 與 **機器學習 (Machine Learning)** 三大部分。

## 競賽背景

- 競賽：Kaggle **Home Credit – Credit Risk Model Stability**（信用違約風險模型穩定性）。
- 任務：依據申請人在申請當下的歷史資料，預測其 **是否違約**（二元分類，`target` ∈ {0, 1}）。
- 資料架構：以 `train_base.parquet`（每個 `case_id` 一筆）為基底，橫向 join 多張 `depth_0 / depth_1 / depth_2` 的靜態與歷史交易資料表。
- 時間軸：以 `WEEK_NUM` 表示資料所屬週次，供時間／群組感知的交叉驗證使用。

## 總覽流程圖

```
[1] 載入套件 & 設定種子 (SEED=0)
        │
[2] 設定路徑 ROOT="data/train"
        │
[3] 讀入 final_columns（最終欄位白名單，約 290 欄）與 cat_cols（類別欄位）
        │
[4] 資料工程（Polars）────────────────────────────┐
      │   read_file / read_files                    │
      │   ├─ Pipeline.set_table_dtypes（依後綴轉型） │
      │   ├─ prune_columns（依 final_columns 精簡） │
      │   ├─ depth 1/2 → group_by(case_id) 聚合    │
      │   └─ shrink（降型省記憶體）                 │
      ▼                                             │
[5] 特徵工程（Polars）                               │
      │   feature_eng()                             │
      │   ├─ 產生 month_decision / weekday_decision │
      │   ├─ 依 case_id 左 join 全部 depth 0/1/2   │
      │   └─ Pipeline.handle_dates（日期→天數差）   │
      ▼                                             │
[6] 轉成 pandas：select(final_columns)              │
      │   → to_pandas()（轉 category）              │
      │   → reduce_mem_usage()（型別縮到最小）       │
      ▼                                             │
[7] 機器學習 ────────────────────────────────────────┘
      │   StratifiedGroupKFold(5) by WEEK_NUM
      │   ├─ CatBoostClassifier（GPU, 每折訓練）
      │   └─ LGBMClassifier（CPU, 每折交替 params1/2）
      │        每折記錄驗證 AUC
      ▼
[8] 集成 VotingModel（10 個模型平均）→ 存 ensemble_model.pkl
      │
[9] 產生 submission.csv（依 max_pmts_year 做年度 score 校正）
```

---


## 一、資料工程 (Data Engineering)

### 1.1 資料來源與讀取

- 資料根目錄：`ROOT = Path("data")`、`TRAIN_DIR = ROOT / "train"`。
- 以 `case_id` 為唯一主鍵，讀取三種深度層級的 parquet：
  - `depth_0`：靜態表 `train_static_cb_0.parquet`、`train_static_0_*.parquet`。
  - `depth_1`：申請人歷史／徵信／稅務等 10 張表（`train_applprev_1_*.parquet`、`train_credit_bureau_*`、`train_person_1`、`train_deposit_1`、`train_debitcard_1`…）。
  - `depth_2`：更深的歷史交易表（`train_credit_bureau_*_2`、`train_applprev_2`、`train_person_2`）。

### 1.2 型別設定 `Pipeline.set_table_dtypes`

依欄位名稱後綴自動判定型別（Polars）：

| 後綴 | 轉型 | 範例 |
|------|------|------|
| `case_id`, `WEEK_NUM`, `num_group1/2` | `Int64` | — |
| `date_decision` | `Date` | — |
| `P`, `A`（數值金額類） | `Float64` | `credamount_770A` |
| `M`（類別文字類） | `String` | — |
| `D`（日期類） | `Date` | `birthdate_574D` |

### 1.3 欄位過濾 `Pipeline.filter_cols`

- 刪除 **缺值率 > 70%** 的欄位。
- 字串欄位：唯一值數 = 1（常數）或 > 200 者刪除。

### 1.4 欄位精簡 `prune_columns` / `final_columns`

- 由 `final_columns` 白名單（約 290 欄）決定最終保留的欄位，其餘在讀取時即丟棄。
- `base_name()` 會先剝除 `max_ / min_ / last_ / first_ / mean_` 等聚合前綴、以及 join 產生的 `_<n>` 尾碼，與白名單比對。
- 讀完後再 `select([col for col in final_columns])` 只保留模型欄位，大幅節省記憶體。

### 1.5 記憶體管理

- `shrink()`：Polars 端降型 — `Float64→Float32`、`Int64→Int32`（保留主鍵與 target 為 Int64）。
- `to_pandas()`：逐欄 `drop_in_place` 搬移，避免 Polars 與 pandas 同時各持一份完整資料；`cat_cols` 在搬移時先 `cast(pl.String)` 再 `cast(pl.Categorical)`，最後轉 pandas 後再設 `category`。
- `reduce_mem_usage()`：pandas 端依實際數值範圍（min/max）把整數與浮點縮到最小可容納型別（int8→int64、float16→float64），`category` 型別跳過。
- 全程頻繁 `del` + `gc.collect()` 釋放記憶體。

---
## 二、特徵工程 (Feature Engineering)

### 2.1 基底時間特徵

- 從 `date_decision` 產生：
  - `month_decision`（決策月份）
  - `weekday_decision`（決策星期幾）

### 2.2 日期欄位處理 `Pipeline.handle_dates`

- 所有後綴 `D` 的日期欄與 `date_decision` 相減，再取 `dt.total_days()` 轉成 **「相對天數差」**，取代原始日期值。
- 之後 drop `date_decision`、`MONTH`。

### 2.3 分組聚合 `Aggregator`（Polars `group_by("case_id")`）

對 depth-1 / depth-2 的歷史表依 `case_id` 聚合，依欄位後綴套用不同統計量：

| 聚合方法 | 後綴 | 統計量 |
|----------|------|--------|
| `num_expr` | `P`, `A`（數值） | `max` + `last` + `mean` |
| `date_expr` | `D`（日期） | `max` + `last` + `mean` |
| `str_expr` | `M`（文字） | `max` + `last` |
| `other_expr` | `T`, `L`（其他） | `max` + `last` |
| `count_expr` | 含 `num_group` 欄 | `max` + `last` |

- 聚合後欄位以 `max_<col>`、`last_<col>`、`mean_<col>` 命名。
- `read_files` 讀完多個分片後 `pl.concat`（`vertical_relaxed`），再 `unique(subset=["case_id"])` 去重。

### 2.4 合併 `feature_eng`

- 將 `df_base` 與所有 depth_0 / depth_1 / depth_2 的聚合結果，依 `case_id` 逐張 **left join**（suffix 為 `_<i>`）。
- 每 join 完一張即把來源表置為 `None` 並 `gc.collect()`，降低峰值記憶體。
- 最後再 `handle_dates` 與 `shrink`。

### 2.5 類別欄位

- `cat_cols` 自 `data/cat_cols.pkl` 載入（約 113 個類別欄位）。
- 轉 pandas 後 `df_train[cat_cols] = df_train[cat_cols].astype(str)` 供後續 LightGBM 以 `category` 使用。

---
## 三、機器學習 (Machine Learning)

### 3.1 資料切分

- `X = df_train.drop(columns=["target","case_id","WEEK_NUM"])`、`y = df_train["target"]`。
- 以 `weeks = df_train["WEEK_NUM"]` 為 groups，使用 **`StratifiedGroupKFold(n_splits=5, shuffle=False)`**，確保同一週的資料不會被切到不同折（時間／群組感知）。

### 3.2 模型一：CatBoost（GPU）

每折訓練一個 `CatBoostClassifier`：

```python
CatBoostClassifier(
    best_model_min_trees = 1200,
    boosting_type = "Plain",
    eval_metric = "AUC",
    iterations = 6000,
    learning_rate = 0.05,
    l2_leaf_reg = 10,
    max_leaves = 64,
    random_seed = 42,
    task_type = "GPU",
    use_best_model = True,
)
```

- 使用 `Pool(X_train, y_train, cat_features=cat_cols)` 直接吃類別欄位。
- 以 `eval_set=val_pool` 進行驗證（`use_best_model` 取最佳模型）。

### 3.3 模型二：LightGBM（CPU）

每折交替使用兩組參數 `params1` / `params2`：

| 參數 | params1 | params2 |
|------|---------|---------|
| boosting_type | gbdt | gbdt |
| extra_trees | True | True |
| learning_rate | 0.05 | 0.03 |
| max_depth | 20 | 16 |
| num_leaves | 64 | 72 |
| l1_regularization | 0.1 | 0.1 |
| l2_regularization | 10 | 10 |
| n_estimators | 2000 | 2000 |
| metric / objective | auc / binary | auc / binary |

- `X_train / X_valid` 的 `cat_cols` 轉回 `category` 型別餵給 LightGBM。
- `callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)]`。

### 3.4 交叉驗證評估

- 每折以 `roc_auc_score(y_valid, y_pred_valid)` 記錄驗證 AUC。
- 分別印出 CatBoost 與 LGBM 的 CV AUC 序列及最大值，做為模型比較基準。

### 3.5 集成 VotingModel

- 自訂 `VotingModel(BaseEstimator, RegressorMixin)`：
  - `predict_proba(X)`：對前 5 個 CatBoost 與後 5 個 LGBM 的 `predict_proba` 取**平均值**（`np.mean`）。
  - `predict(X)`：各模型 `predict` 的平均。
- 共 10 個模型（5 折 CatBoost + 5 折 LGBM）集成。
- 以 `joblib.dump(model, "ensemble_model.pkl")` 儲存模型。

### 3.6 產生提交檔 submission.csv

```python
df_train = df_train.set_index("case_id")
df_train['score'] = model.predict_proba(df_train.loc[:, df_train.columns != "WEEK_NUM"])[:, 1]

# 依申請年度做 score 校正（減去一定值後 clip(0)）
mask = df_train["max_pmts_year_1139T"] == 2020
df_train.loc[mask, 'score'] = (df_train.loc[mask, 'score'] - 0.07).clip(0)
mask = df_train["max_pmts_year_1139T"] == 2021
df_train.loc[mask, 'score'] = (df_train.loc[mask, 'score'] - 0.06).clip(0)
mask = df_train["max_pmts_year_1139T"] == 2022
df_train.loc[mask, 'score'] = (df_train.loc[mask, 'score'] - 0.02).clip(0)

df_subm = pd.read_csv(ROOT / "sample_submission.csv").set_index("case_id")
df_subm["score"] = df_train['score']
df_subm.to_csv("submission.csv")
```

- 先對全部案例預測違約機率 `score`。
- 依 `max_pmts_year_1139T`（最大繳款年份）分別扣 0.07 / 0.06 / 0.02，再 `clip(0)`，做**年度趨勢校正**。
- 最後以 `sample_submission.csv` 的 `case_id` 順序輸出 `submission.csv`。

---

## 環境與可重現性

- 執行環境：Kaggle Notebook（GPU：NVIDIA T4）、Python 3.13。
- 主要套件：`polars`、`pandas`、`numpy`、`lightgbm`、`catboost`、`scikit-learn`。
- 可重現性：`SEED = 0`，透過 `seed_it_all()` 設定 `PYTHONHASHSEED`、`random.seed`、`np.random.seed`；LightGBM / CatBoost 亦固定 `random_state / random_seed = 42`。

---

## 附錄：Cell 對照表

| Cell | 內容 | 階段 |
|------|------|------|
| 1 | 匯入套件（polars/pandas/lightgbm/catboost/sklearn…） | 前置 |
| 2 | `seed_it_all(SEED=0)` 設定隨機種子 | 前置 |
| 3 | 定義 `Pipeline`（set_table_dtypes / handle_dates / filter_cols） | 資料工程 |
| 4 | 定義 `Aggregator`（五組聚合 expression）與 `read_file(s)` / `shrink` | 資料工程 |
| 5 | 設定 `ROOT / TRAIN_DIR` 路徑 | 前置 |
| 6 | 定義 `final_columns`（約 290 欄白名單） | 資料工程 |
| 7 | 讀取 `cat_cols.pkl`（類別欄位清單） | 資料工程 |
| 8 | 建立 `data_store`（讀取並聚合 depth 0/1/2 全部資料表） | 資料工程 |
| 9 | `feature_eng()` 合併全部表 → 產生 `df_train` | 特徵工程 |
| 10 | `to_pandas()` 轉 pandas + `reduce_mem_usage()` 降記憶體 | 資料工程 |
| 11 | `cat_cols` 轉為 `str` | 資料工程 |
| 12 | 定義 `VotingModel`（集成平均） | 機器學習 |
| 13 | 定義 CV、`params1`/`params2`，5 折訓練 CatBoost + LGBM | 機器學習 |
| 14 | 檢視模型 feature_names | 附註 |
| 15 | 標題：Submission | — |
| 16-19 | 檢視 `df_train` 等 | 附註 |
| 20 | 產生 `submission.csv`（含年度 score 校正） | 機器學習 |