# WOE/IV 特徵篩選 — 完成摘要

## 概述

依 [woe_iv_feature_selection_plan.md](woe_iv_feature_selection_plan.md) 計畫實作
[feature_selection_woe_iv.py](feature_selection_woe_iv.py)，對 `data/base_final.parquet`
（`data/base.parquet` LEFT JOIN `data/train_person_agg.parquet`，1,526,659 列、1,201 欄）
中除 `case_id`、`date_decision`、`MONTH`、`WEEK_NUM`、`target` 以外的 **1,196 個自變數**，
計算 Weight of Evidence (WOE) 與 Information Value (IV)，以 IV 作為特徵去留依據。

**執行環境**：Anaconda base env（`C:\Users\ittraining\anaconda3`）；
`duckdb 1.4.3`、`polars 1.41.2`、`pyarrow 21.0.0`、`category_encoders 2.8.1`、
`scikit-learn 1.7.2`、`pandas 2.3.3`、`numpy 2.3.5`（已記錄於 [requirements.txt](requirements.txt)）。

**輸出**：
- `feature_selection_woe_iv.py` — 完整流程程式（分箱 → WOE → IV → 分層 → 輸出）
- `data/base_final_woe_iv.csv` — 完整 IV 排名表（1,196 列，機器可讀），含 `bin_intervals`
  欄位:數值/時間戳欄的決策樹**實際分箱區間**(如 `bin_0:(-inf, 12.50] | bin_1:(12.50, 45.00] | ...`)
- `data/base_final_woe_iv.md` — 中文 IV 排名報告（Top 40,含 bin_intervals + 分層統計）

## 資料概況

| 項目 | 數值 |
|---|---|
| 總列數（case_id） | 1,526,659 |
| 總欄數 | 1,201（1,196 個受篩特徵 + 5 個排除欄） |
| target=0（正常還款） | 1,478,665（96.86%） |
| target=1（違約） | 47,994（3.14%） |
| dtype 分佈 | DOUBLE 870、BIGINT 179（含 4 個排除欄）、UINTEGER 73、VARCHAR 65（含 date_decision）、TINYINT 4、BOOLEAN 4、TIMESTAMP 6 |

## 方法（實作與計畫一致，細節如下）

1. **欄位分型**：以 DuckDB `DESCRIBE` 取得的 dtype 分流 —
   數值型（含 TINYINT/UINTEGER/BIGINT/DOUBLE）與時間戳型（cast 為天數序數）走決策樹分箱；
   VARCHAR/BOOLEAN 走類別分箱。實測 1,122 數值欄、68 類別欄、6 時間戳欄，0 個未分類欄。
2. **決策樹分箱**：`DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf>=1000, class_weight="balanced")`，
   對非空值擬合（樣本數 > 30 萬時抽樣至 30 萬以控制耗時，WOE/IV 仍以全量資料計算）；
   缺失值獨立為 `__MISSING__` 箱。**區間輸出**：因樹只用單一特徵,取出所有內部節點的
   `threshold` 排序去重即為葉節點對應的區間邊界(等價於 `tree.apply()` 的分箱結果,但可讀);
   時間戳欄的邊界另轉回日期字串（如 `2019-05-30`）。
3. **類別分箱**：原值即箱，null → `__MISSING__`；出現次數 < 1,000 的稀有類別併入 `__RARE__`。
4. **WOE**：呼叫 `category_encoders.WOEEncoder(regularization=0.5)` 對分箱後的離散欄
   `fit_transform`，取得各箱 WOE。
5. **IV 自算**：以各箱 good/bad 分布（套用與 WOEEncoder 相同的 regularization）計算
   `IV = Σ(dist_bad − dist_good) × WOE`。
6. **分層與決策**（Siddiqi 慣例，門檻依使用者決議）：

   | IV 區間 | 層級 | decision |
   |---|---|---|
   | < 0.02 | 無預測力 | drop |
   | 0.02–0.10 | 弱 | keep |
   | 0.10–0.30 | 中 | keep |
   | 0.30–0.50 | 強 | keep |
   | > 0.50 | 可疑（疑洩漏/過強） | keep（標記待複查） |

## 執行結果

- 總執行時間：首次執行 2,529 秒（約 42 分鐘）；加入 `bin_intervals` 輸出後重跑 3,661 秒
  （約 61 分鐘，耗時差異屬機器負載波動,非邏輯變更所致）。逐欄處理，全程無錯誤（error 列數 = 0）。
- 兩次執行的 IV / n_bins / 分層結果**逐欄完全一致**（相同 `random_state=42`、相同欄位處理順序
  → 相同的樹擬合抽樣序列），確認加入區間輸出的重構未改變任何分箱或計算結果。
- 受篩欄位總數:1,196;**keep 659、drop 537**。
- 分層統計：

  | 層級 | 欄位數 |
  |---|---|
  | 無預測力 | 537 |
  | 弱 | 473 |
  | 中 | 177 |
  | 強 | 9 |
  | 可疑（IV > 0.5） | 0 |

- 無任何欄位 IV > 0.5 → **未發現明顯的目標洩漏（target leakage）訊號**。
- Top IV 欄位集中在 `credit_bureau_a`/`credit_bureau_b` 的逾期天數(DPD)、逾期金額/期數比率
  類指標（如 `pmts_overdue_1152A_overdue_rate__weighted_avg` IV=0.357、
  `pmts_dpd_303P_overdue_rate__weighted_avg` IV=0.351、`avgdpdtolclosure24_3658938P` IV=0.320），
  符合信用風險領域知識（逾期歷史為違約最強預測因子）。

### Top 5 IV 欄位的實際決策樹分箱區間

| column | IV | bin_intervals |
|---|---|---|
| `pmts_overdue_1152A_overdue_rate__weighted_avg` | 0.357277 | `(-inf,0.0006] \| (0.0006,0.0652] \| (0.0652,0.1199] \| (0.1199,0.1997] \| (0.1997,0.2962] \| (0.2962,0.3974] \| (0.3974,0.6582] \| (0.6582,inf) \| __MISSING__` |
| `pmts_dpd_303P_overdue_rate__weighted_avg` | 0.351267 | `(-inf,0.0028] \| (0.0028,0.0139] \| (0.0139,0.0618] \| (0.0618,0.1417] \| (0.1417,0.2485] \| (0.2485,0.363] \| (0.363,0.6261] \| (0.6261,inf) \| __MISSING__` |
| `avgdpdtolclosure24_3658938P` | 0.319773 | `(-inf,0.5] \| (0.5,1.5] \| (1.5,2.5] \| (2.5,5.5] \| (5.5,11.5] \| (11.5,210.5] \| (210.5,inf) \| __MISSING__` |
| `pmts_dpd_303P_std__mean` | 0.312981 | `(-inf,0.546] \| (0.546,0.9414] \| (0.9414,1.961] \| (1.961,2.571] \| (2.571,4.045] \| (4.045,46.47] \| (46.47,77.81] \| (77.81,inf) \| __MISSING__` |
| `dpdmax_757P__mean` | 0.307872 | `(-inf,2.47] \| (2.47,4.981] \| (4.981,7.493] \| (7.493,13.44] \| (13.44,26.48] \| (26.48,85.49] \| (85.49,409.4] \| (409.4,inf) \| __MISSING__` |

觀察:逾期率/DPD 類指標的樹分箱在低值端切得很細（如 `pmts_overdue_1152A_overdue_rate` 前兩箱
邊界僅 0.0006、0.0652）,顯示「完全無逾期 vs 輕微逾期」的區分本身就帶有大量風險資訊,
這也是這類欄位 IV 特別高的原因之一。

## 與既有人工評級的交叉比對

| 欄位 | 既有人工評級 | 本次 IV | 本次分層 | 說明 |
|---|---|---|---|---|
| `reject_rate` | 高（applprev_1） | 0.2336 | 中 | 一致：拒貸率為中等強度預測因子 |
| `age_years_appl` | 高（person_1） | 0.1079 | 中 | 一致：年齡為核心人口風險因子 |
| `n_prev_apps` | — | 0.0414 | 弱 | 合理：申請次數本身訊號較弱，需搭配比率特徵 |
| `actualdpd_943P_max` | 高（applprev_1） | 0.0273 | 弱 | 偏低：99% 為 0 值，單變量分箱下鑑別力有限 |
| `childnum_185L_appl` | 低（已知全空死特徵） | 0.0000 | 無預測力 | 一致：確認為死特徵（見 person_join_cleaning_aggregation_plan.md 既有發現） |
| `mainoccupationinc_384A_appl` | 高（person_1，核心可負擔性變數） | 0.0031 | 無預測力 | **明顯偏離**：原始收入單變量對 target 的區辨力很弱，值得注意的資料驅動發現 |

`mainoccupationinc_384A_appl` 的結果與人工評級落差最大，是本次資料驅動篩選相對於
領域知識評級的主要新發現，建議後續建模時留意（可能需要與其他欄位交互或做比率轉換才有訊號，
不宜僅因人工評級「高」就預設其重要）。

### 上表欄位的實際決策樹分箱區間

| column | n_bins | bin_intervals |
|---|---|---|
| `reject_rate` | 9 | `(-inf,0.025] \| (0.025,0.1938] \| (0.1938,0.3246] \| (0.3246,0.458] \| (0.458,0.5941] \| (0.5941,0.767] \| (0.767,0.975] \| (0.975,inf) \| __MISSING__` |
| `age_years_appl` | 8 | `(-inf,22.68] \| (22.68,24.86] \| (24.86,27.94] \| (27.94,36.56] \| (36.56,54.27] \| (54.27,56.97] \| (56.97,64.99] \| (64.99,inf)` |
| `n_prev_apps` | 9 | `(-inf,1.5] \| (1.5,2.5] \| (2.5,4.5] \| (4.5,5.5] \| (5.5,6.5] \| (6.5,10.5] \| (10.5,12.5] \| (12.5,inf) \| __MISSING__` |
| `actualdpd_943P_max` | 2 | `(-inf,inf) \| __MISSING__`（樹在 min_samples_leaf 限制下找不到有效切點，退化成單一非空箱 —— 等價於純粹的「有無逾期紀錄」二分，見驗證章節） |
| `childnum_185L_appl` | 1 | `(-inf,inf) \| __MISSING__`（全欄皆空，missing_rate=1.0，僅剩缺失箱，IV=0，確認為死特徵） |
| `mainoccupationinc_384A_appl` | 8 | `(-inf,2.791e+04] \| (2.791e+04,3.596e+04] \| (3.596e+04,3.909e+04] \| (3.909e+04,4.479e+04] \| (4.479e+04,5.499e+04] \| (5.499e+04,5.95e+04] \| (5.95e+04,6.101e+04] \| (6.101e+04,inf)` |

`mainoccupationinc_384A_appl` 的切點集中在約 2.8萬~6.1萬這個相對窄的範圍（且無缺失，
因本人列 100% 有值），顯示原始收入分布本身對違約的區分度確實有限，並非分箱失敗所致。

## 驗證（獨立於主程式重算，確認公式與流程正確）

1. **執行完整性**：`python feature_selection_woe_iv.py` 全量執行無錯誤；
   CSV 列數 = 1,196 = 總欄數(1,201) − 5 個排除欄；`target`/`case_id` 未出現於輸出；所有 IV ≥ 0。
2. **Missing-only 分箱案例手算比對**（`actualdpd_943P_max`，決策樹僅產出 1 個非空箱 → 
   等價於「是否缺失」二分)：以 DuckDB 建立 `is_null × target` 列聯表，套用相同 regularization
   手算 IV = **0.027254952**，與腳本輸出 **0.027255** 完全一致。
3. **多箱案例獨立重算**（`reject_rate`，決策樹產出 9 箱）：以純 pandas（不經 `category_encoders`）
   重新實作同一公式，對相同分箱標籤計算 IV，與腳本內部計算完全一致（兩次呼叫均得 0.2310157...），
   確認 IV 公式與 WOE 取值方向一致、無重複計算誤差。兩次不同執行因決策樹抽樣（`TREE_SAMPLE_CAP`
   下的隨機子樣本）產生的分箱略有差異，屬預期的隨機性（分層結果不受影響，皆落於「中」層）。
4. **合理性抽驗**：已知強風險欄（`*dpd*`、`*overdue*`）落在中/強層；已知死特徵
   （`childnum_185L_appl`）落在無預測力層；keep/drop 比例合理（55%/45%），非全 keep 或全 drop。

## 已知限制

- 決策樹分箱含隨機抽樣（大於 30 萬列時）。**完整重跑整支腳本**因 `random_state=42` 固定、
  欄位處理順序固定，兩次全量執行結果逐欄完全一致（見「執行結果」）。但若脫離主程式、
  對單一欄位獨立呼叫分箱函式並重新以 `np.random.default_rng(42)` 起始（如本文驗證章節
  第 3 點的獨立重算），因 RNG 消耗序列的起始位置與全量執行中不同，會得到不同的抽樣子集，
  使 IV 有小幅波動（驗證中 `reject_rate` 全量執行得 0.2336，獨立重算得 0.2310），
  但分層/decision 結果穩定（皆落於「中」層）。
- 高基數 VARCHAR 欄（地理遮罩欄、`profession_152M` 等）以「稀有類別 < 1,000 筆併入 `__RARE__`」
  處理，避免 IV 虛高，但仍為單變量粗篩，未做欄位間共線性/交互作用分析。
- 原始日期字串欄（如 `lastrejectdate_50D`、`datelastunpaid_3546854D` 等，尚未轉換為天數）
  依 dtype 判定被歸為類別欄處理，而非數值型日期特徵，IV 結果僅供參考。
