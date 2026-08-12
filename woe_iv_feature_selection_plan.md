# WOE/IV 特徵篩選計畫（base_final.parquet）

## Context

專案已把 Home Credit 各來源表清洗、聚合並 join 成寬表 `data/base_final.parquet`
（由 [train_person_join_base.py](train_person_join_base.py) 產出 = `data/base.parquet` LEFT JOIN
`data/train_person_agg.parquet`，以 `case_id` 為鍵，約 1.5M 列 × ~900 欄）。
目標欄為 `target`（1=違約、0=正常還款，見 [Home_Credit_2024_欄位整理共用 - 最終BASS表.csv](Home_Credit_2024_欄位整理共用 - 最終BASS表.csv) 第 7 列）。

至今的欄位粗篩（[data/applprev_1_importance.md](data/applprev_1_importance.md)、
[data/person_1_importance.md](data/person_1_importance.md)）是**領域知識人工評級**。
本次要換成**資料驅動的單變量篩選**：對每個自變數計算 Weight of Evidence (WOE) 與
Information Value (IV)，以 IV 作為保留/剔除特徵的依據。這是信用評分卡的標準做法，
能在建模前把 ~900 欄壓縮成一份帶 IV 排名與去留決策的清單。

**需求要點（已與使用者確認）**
- WOE 轉換使用 `category_encoders`（`WOEEncoder`）。
- 連續型欄位以**決策樹分箱**（target-aware，sklearn，不引入 optbinning）後再算 WOE。
- 篩選門檻：**保留 IV ≥ 0.02**（僅剔除「無預測力」層，不設上限、不因 IV 過高而剔除，但仍標記可疑層供人工複查）。
- 產出：**IV 排名表**（CSV 供機器讀取 + Markdown 報告，風格沿用既有 `*_importance.md`）。

**關鍵技術事實**：`category_encoders.WOEEncoder` 只吃離散欄位、且**不輸出 IV** —— 它只給
每個類別的 WOE 對照。因此本方案必須：(1) 先把數值欄分箱成離散碼，(2) 用 `WOEEncoder`
產生各欄 WOE 對照（滿足「以 category_encoders 轉換每個自變數」），(3) 另以各箱 good/bad
分布**自行計算 IV**。repo 目前無任何 WOE/IV 程式，且 `category_encoders` 尚未列入相依套件。

## 產出檔案

- **新增** `feature_selection_woe_iv.py`（repo 根目錄，常數 `SRC`/`OUT` 與 `main()` 進入點沿用
  [data_preprocessing_2.py](data_preprocessing_2.py) 風格）。
- **新增** `data/base_final_woe_iv.csv` —— 完整 IV 排名表（機器可讀）。
- **新增** `data/base_final_woe_iv.md` —— 中文 IV 排名報告，格式對齊 `data/applprev_1_importance.md`。

## 前置相依

repo 無 `requirements.txt`。本腳本需 `category_encoders`、`scikit-learn`、`pandas`
（`polars`、`pyarrow` 已在用）。安裝：`pip install category_encoders scikit-learn pandas`。
建議一併新增 `requirements.txt` 記錄相依（可選）。

## 方法設計

### 步驟 1：欄位分型（依 dtype，不靠欄名硬解析）

用 `pl.scan_parquet(SRC).collect_schema()` 取得型別後：
- **排除欄**：`case_id`、`date_decision`、`MONTH`、`WEEK_NUM`、`target`（非特徵/為目標）。
- **數值欄**：整數/浮點 dtype（含 `__mean/__max/__min/__std/__sum/__null_rate/__non_null_count`、
  各種比率與計數、`has_*`/`is_*` 0/1 旗標）→ 走決策樹分箱。
- **類別欄**：`Utf8`/`Categorical`（`__mode`、M 尾遮罩欄、`_appl` 類別欄、`last_status`、
  `card_status_last` 等）→ 直接當離散箱。

### 步驟 2：分箱（產出各欄離散碼）

- **數值欄（決策樹分箱，target-aware）**：對非空列以
  `DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=max(1000, int(0.02*n)))`
  擬合 `X=col、y=target`，以 `tree.apply(X)` 的葉節點 id 當箱碼；
  空值（NaN）獨立成 `"__MISSING__"` 箱（缺失率本身即訊號，沿用 repo 一貫立場）。
  `min_samples_leaf` 確保每箱樣本足夠、WOE 穩定。可對「擬合用」資料抽樣（上限如 30 萬列）
  以加速，但 **WOE/IV 統計仍以全量計算**。
- **類別欄**：原值即箱；NaN → `"__MISSING__"`；
  高基數欄（如 `*_district_*M`、`*_zipcode_*M`、`profession_*M`，基數達數千）為避免 IV 虛高，
  將出現次數低於門檻（如 < `min_samples_leaf`）的稀有類別併入 `"__RARE__"`（門檻與併箱規則寫入報告）。

### 步驟 3：WOE（category_encoders）+ IV（自算）

逐欄（或小批次，控記憶體）：
1. 以分箱後的離散欄建 pandas 欄，`ce.WOEEncoder(cols=[c], regularization=0.5).fit(df[[c]], y)`，
   由 `enc.mapping[c]` 取得**各箱 WOE**（此即「以 category_encoders 轉換每個自變數」）。
2. 由各箱 good（target=0）/ bad（target=1）計數求分布：
   `dist_good = good/total_good`、`dist_bad = bad/total_bad`（加 `regularization` 平滑避免除零）。
3. **IV = Σ_bin (dist_bad − dist_good) × WOE_bin**；WOE 取自 category_encoders mapping
   （其定義 `WOE = ln(dist_bad/dist_good)`，target=1 為 bad，方向與 IV 一致，IV ≥ 0）。
   驗證時抽數欄以手算 `ln(dist_bad/dist_good)` 對照 mapping，確認方向/平滑一致。

記憶體策略：~900 欄 × 1.5M 列不需同時展開稠密矩陣 —— 以 polars 逐欄/分批取數、
只保留各欄的箱計數表（極小）即可；`WOEEncoder` 逐欄擬合，記憶體有界。

### 步驟 4：分層與去留決策（Siddiqi 慣例）

| IV 區間 | 層級 | decision |
|---|---|---|
| < 0.02 | 無預測力 | **drop** |
| 0.02 – 0.10 | 弱 | keep |
| 0.10 – 0.30 | 中 | keep |
| 0.30 – 0.50 | 強 | keep |
| > 0.50 | 可疑（疑洩漏/過強） | keep，但標記待人工複查 |

決策規則：**IV ≥ 0.02 即 keep**（無上限）。常數/全空/無法切分的欄 → n_bins=1、IV=0 → drop。

### 步驟 5：輸出

- `data/base_final_woe_iv.csv`：欄位 `column, dtype, n_bins, missing_rate, IV, tier, decision`，依 IV 遞減排序。
- `data/base_final_woe_iv.md`：中文報告（Context/方法/門檻表/Top 排名摘要/keep-drop 統計/可疑欄清單），
  格式對齊 `data/applprev_1_importance.md`。
- stdout 印出：篩選欄數、keep/drop 各層計數、Top-N IV 欄位。

## 驗證方式

1. 執行 `python feature_selection_woe_iv.py`，無錯誤；CSV 列數 = 受篩欄數（≈ 總欄數 − 5 個排除欄）。
2. `target` 不在輸出；所有 IV ≥ 0。
3. 合理性抽驗（DuckDB）：已知強風險欄（`*overdue*`、`*dpd*`、`outstandingdebt_*`、
   `reject_rate` 等）落在中/強層；近常數/近全空欄落在無預測力層。
4. 手算對照：挑 2~3 欄，用 DuckDB 建「箱 × target」列聯表手算 IV，與腳本結果比對（容差內一致）；
   同時抽驗 category_encoders 的 WOE 與手算 `ln(dist_bad/dist_good)` 一致。
5. keep/drop 統計合理（不會全 keep 或全 drop）。

---

實際執行結果與驗證細節見 [woe_iv_feature_selection_summary.md](woe_iv_feature_selection_summary.md)。
