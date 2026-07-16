# applprev_2_agg 欄位二次聚合 — 完成摘要

## 概述

依 `applprev_1_join_cleaning_aggregation_plan.md` 的增量計畫實作 [data_preprocessing_2.py](data_preprocessing_2.py),將 `train_applprev_join.parquet` 中來自 `train_applprev_2_agg.parquet` 的 15 個指標欄(3 個原始類別欄 × 5 個指標),從「每筆先前申請」`(case_id, num_group1)` 層級**二次聚合**到 `case_id` 層級。

**輸出**:`data/train_applprev_1_agg.parquet`,欄位數 57 → **77**(1,221,522 列不變,等於 distinct case_id 數)。

## 三個來源欄位的策略與新特徵

依各欄的訊號強度採不同策略,避免產出無變異的死特徵。

| 來源欄位 | 說明 | 策略 | 新特徵(共 20 個) |
|---|---|---|---|
| **conts_type_509L** | 先前申請的聯絡方式類型(訊號最豐富) | 完整指標組 + 跨申請眾數 | `conts_type_509L_mode_last`、`_mode_ratio_mean`、`_mode_ratio_min`、`_n_unique_max`、`_n_unique_mean`、`_entropy_max`、`_entropy_mean`、`_null_count_sum`;跨申請眾數 `_mode_mode`、`_mode_mode_ratio`、`_mode_n_unique`、`_mode_entropy` |
| **cacccardblochreas_147M** | 卡片凍結原因(99.9% 為佔位值,無變異) | 特化為稀有事件旗標 | `has_card_block_any`、`n_card_block_apps`、`cacccardblochreas_147M_null_count_sum` |
| **credacc_cards_status_52L** | 先前信用帳戶卡片狀態(95% 結構性缺失) | 持卡旗標 + 負面狀態旗標 | `has_card_any`、`n_card_apps`、`card_status_last`、`card_cancelled_or_blocked_any`、`credacc_cards_status_52L_null_count_sum` |

### 指標選用理由(重點)

- **conts_type_509L**:`n_unique`(申請內聯絡方式多樣性)與跨申請 `mode_n_unique`(歷史換過幾種主要聯絡方式)是不同層次的完整度訊號,兩者並存;`mode_ratio_mean` 反映集中度,`entropy` 與 `n_unique` 互補(前者衡量分布是否均勻,後者只算種類數);`null_count_sum` 為資料完整度,資訊不透明本身可能是風險訊號。
- **cacccardblochreas_147M**:mode/mode_ratio/entropy/n_unique 在 99.9% 的列上無變異,做統計會得到死特徵。改為「曾出現真實卡片凍結原因」的事件旗標(強負面信用事件),稀有(0.38% case)但對樹模型有效。
- **credacc_cards_status_52L**:95% 整組缺失下,「有無卡」本身即特徵;狀態類統計只在有卡申請上有意義,故只取最近狀態與「曾取消/凍結」旗標,不做 ratio/entropy(有值樣本 n_unique 94.6%=1,無變異)。

### 設計原則

二次聚合的 `mean` 為「**每筆申請等權**」而非「每筆聯絡紀錄等權」——case 層級特徵以「申請」為分析單位,申請內紀錄多寡不影響權重。

## 缺失值處理

- `_mode/_mode_ratio/_n_unique/_entropy` 的 NULL = 該筆申請整組無紀錄(結構性缺失)→ **不補值**,聚合統計天然忽略;`has_*` 旗標已顯式編碼「有無紀錄」。
- 3 個 `_null_count` 欄各僅 1 列 NULL → 補 0。

## 驗證結果(DuckDB 抽驗)

| 檢查項 | 結果 |
|---|---|
| 輸出列數 | 1,221,522 = distinct case_id ✓ |
| 旗標盛行率 vs 探索數據 | `has_card_block_any`=4,674、`has_card_any`=251,163、`card_cancelled_or_blocked_any`=131,703 **完全一致** ✓ |
| 旗標值域 | 三個旗標欄皆僅 {0,1} ✓ |
| conts_type 範圍 | `mode_ratio_mean` ∈ [0.2, 1.0];`n_unique_max ≥ n_unique_mean`(0 筆違反) ✓ |
| 手工比對 | 2 個 case_id 的 `card_status_last`(最近一筆非空狀態)與 `null_count_sum` 皆正確 ✓ |
| 既有欄位不變 | `reject_rate` 與獨立重算 **0 筆不符**,第一階段 57 欄未受影響 ✓ |

## 實作備註

- 重構 `build_cat_agg`,加入 `drop_nulls` 與 `placeholder` 參數:同時服務 M 欄(保留 `non_placeholder_ratio`)與 conts_type 的跨申請眾數(L 欄無佔位值,`placeholder=False` 略過恆為 1 的死特徵)。對原有 M 欄輸出無影響(M 欄無 NULL)。
- 聚合前已 `sort(["case_id", "creationdate_885D"])`,故 `last()` / `card_status_last` 皆取最近一次申請。
