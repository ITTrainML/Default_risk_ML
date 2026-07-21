# VIF 多重共線性分析計畫（data/final_feature.csv）

## Context

先前已完成 `data/base_final.parquet` 的 WOE/IV 單變量特徵篩選
（見 [woe_iv_feature_selection_summary.md](woe_iv_feature_selection_summary.md)、
[feature_selection_woe_iv.py](feature_selection_woe_iv.py)）。
團隊成員各自以不同方法（AUC+KS、Catboost、EBM、GBM、TOP150 入選-1~5、以及本專案的 WOE）
挑選候選特徵，彙整成 `data/final_feature.csv`：**195 個不重複欄位**（10 個來自我們的 WOE 篩選）。

**目前發現**：
- 195 欄中有 194 欄確實存在於 `base_final.parquet` schema；唯一例外是 `date_decision`
  （決策日期字串），這是專案一貫排除的中繼欄（如 `case_id`/`target`），不應當作模型特徵。
- dtype 組成：157 DOUBLE、22 VARCHAR（類別欄,含 `date_decision`）、9 BIGINT、6 UINTEGER、1 BOOLEAN。
- 194 個候選欄（排除 date_decision）皆無零變異/全常數風險（交叉比對既有 IV 表:無
  `n_bins=1 且 missing_rate=0` 的欄位）；其中 10 欄在先前 WOE/IV 篩選中屬「drop」層級
  （IV<0.02),但因其他方法(如 Catboost/GBM)仍選入,予以保留待 VIF 分析,不在此階段剔除。

**目標**：在最終決定進入模型的特徵集合前,以 **VIF (Variance Inflation Factor)** 檢查這
194 個候選特徵間的多重共線性 —— 這對後續若採 WOE-based 邏輯迴歸評分卡尤其重要
（共線性會使係數不穩定、正負號反轉、標準誤失真)。

**需求要點(已與使用者確認)**:
- **排除 `date_decision`**(非特徵中繼欄),分析 194 個候選特徵。
- **全部 194 欄先以 WOE 轉換**(數值欄與類別欄一視同仁),VIF 在 WOE 數值矩陣上計算
  ——這與之後若採 WOE 評分卡的實際建模輸入一致,也是類別欄唯一可行的數值化方式。
  重用 [feature_selection_woe_iv.py](feature_selection_woe_iv.py)
  已驗證的分箱(決策樹分箱 / 類別稀有併箱)與 `category_encoders.WOEEncoder` 轉換邏輯。
- **僅輸出排名報告,不自動剔除**(report-only):列出每欄 VIF 與分層,由使用者後續人工決定去留。
- **取樣 50 萬列**(固定 random_state,所有欄位共用同一組列樣本以保留欄位間相關結構)
  而非全量 150 萬列 —— 因目前可用記憶體有限(約 5GB)且 VIF 統計量在此樣本數下已足夠穩定。

## 產出檔案

- **新增** `feature_selection_vif.py`(repo 根目錄)—— import 並重用
  `feature_selection_woe_iv.py` 的 `fit_tree_thresholds`/`apply_thresholds`/`cat_bin_labels`
  等分箱函式,避免重複實作。
- **新增** `data/final_feature_vif.csv` —— 194 欄完整 VIF 排名表(機器可讀)。
- **新增** `data/final_feature_vif.md` —— 中文 VIF 報告,格式對齊 `data/base_final_woe_iv.md`。
- **新增** `vif_multicollinearity_summary.md`(repo 根目錄)—— 完成摘要,格式對齊既有
  `woe_iv_feature_selection_summary.md`。

## 方法設計

### 步驟 1:讀取候選欄位清單

解析 `data/final_feature.csv` 的「欄位名稱」欄,去重取得 195 筆,**剔除 `date_decision`**,
得 194 個候選特徵。與 `data/base_final_woe_iv.csv` 交叉比對確認皆存在且無零變異風險
(已於探索階段驗證,見上)。

### 步驟 2:固定列樣本讀取(所有欄位共用)

以 `polars`/`duckdb` 從 `base_final.parquet` **一次性**讀出 194 個候選欄 + `target`,
取固定 random_state(如 42)的 50 萬列樣本(如 `df.sample(n=500_000, seed=42)` 或
DuckDB `USING SAMPLE 500000 (bernoulli, 42)` 等可重現寫法)。
**關鍵**:所有欄位必須用同一批列,不可逐欄各自取樣,否則會破壞欄位間的相關結構,VIF 失真。

### 步驟 3:194 欄全部 WOE 轉換(重用既有分箱邏輯)

對樣本中的每一欄:
- 數值/時間戳欄:呼叫 `feature_selection_woe_iv.fit_tree_thresholds` + `apply_thresholds`
  分箱(決策樹,`max_leaf_nodes=8`、`min_samples_leaf>=1000`,缺失獨立成 `__MISSING__` 箱)。
- 類別欄:呼叫 `cat_bin_labels`(稀有類別 `< 1000` 併入 `__RARE__`)。
- 分箱後以 `category_encoders.WOEEncoder(regularization=0.5)` 轉換為該列的 WOE 值
  (與既有 IV 腳本相同函式/參數,確保方法論一致)。

組成 500,000 × 194 的**全數值 WOE 矩陣**(無缺失,因缺失已編碼為 `__MISSING__` 箱的 WOE 值)。

### 步驟 4:VIF 計算(相關矩陣求逆,快速且與迴歸法等價)

1. 計算 194×194 皮爾森相關矩陣(`numpy.corrcoef`,對標準化變數而言,相關矩陣求逆的對角線
   數學上等價於逐欄對其餘欄回歸的 VIF)。
2. 以 `numpy.linalg.pinv`(虛擬逆,避免近奇異矩陣導致求逆失敗)取得逆矩陣,取對角線即為
   194 個 VIF 值。近完全共線的欄位 VIF 會非常大,以上限值(如 1e6)標記為
   「近完全共線」,附註說明,不讓數值失控。
3. **獨立驗證**:挑選 5 個涵蓋不同 VIF 量級的欄位,另以 `statsmodels.stats.outliers_influence
   .variance_inflation_factor`(逐欄 OLS 回歸法,VIF 的原始定義)重新計算,比對兩法結果一致
   (容差內),確認相關矩陣求逆法正確。

### 步驟 5:分層與輸出(report-only)

| VIF 區間 | 層級 |
|---|---|
| < 5 | 低(可接受) |
| 5 – 10 | 中(留意) |
| > 10 | 高(建議剔除) |

不自動剔除欄位。輸出:
- `data/final_feature_vif.csv`:欄位 `column, source(來源), VIF, tier`,依 VIF 遞減排序。
- `data/final_feature_vif.md`:Context / 方法 / 分層門檻表 / 完整 194 欄排名 / 高 VIF(>10)
  清單摘要,格式對齊 `data/base_final_woe_iv.md`。
- stdout 印出:各層級欄位數、Top-N 高 VIF 欄位。

## 驗證方式

1. 執行 `python feature_selection_vif.py`,無錯誤;輸出列數 = 194。
2. 所有 VIF ≥ 1(數學下界)。
3. 挑 5 欄以 `statsmodels.variance_inflation_factor` 獨立重算,與相關矩陣法結果一致
   (容差內)。
4. 合理性抽驗:已知同源不同視窗的工程特徵(如 `pmts_dpd_303P_mean__mean` 與
   `pmts_dpd_303P_recent12_mean__mean`/`_recent6_mean__mean`/`_recent3_mean__mean`,
   同一序列的不同滾動窗口統計量)應呈現偏高 VIF,驗證分析確實捕捉到共線性。
5. 分層結果不應全部落於單一層級(低/中/高需有分布)。

---

實際執行結果與驗證細節見 [vif_multicollinearity_summary.md](vif_multicollinearity_summary.md)。
