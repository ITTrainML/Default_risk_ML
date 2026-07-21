# VIF 多重共線性分析 — 完成摘要

## 概述

依 [vif_multicollinearity_plan.md](vif_multicollinearity_plan.md) 計畫實作
[feature_selection_vif.py](feature_selection_vif.py)，對 `data/final_feature.csv`
（團隊以 AUC+KS、Catboost、EBM、GBM、TOP150 入選-1~5、WOE 等 9 種方法彙整的
195 個候選特徵）計算 VIF (Variance Inflation Factor)，檢查多重共線性。

**輸出**：
- `feature_selection_vif.py` — 完整流程程式（取樣 → WOE 轉換 → VIF 計算 → 分層 → 輸出）
- `data/final_feature_vif.csv` — 194 欄完整 VIF 排名表（機器可讀）
- `data/final_feature_vif.md` — 中文 VIF 報告（完整排名 + statsmodels 交叉驗證 + 分層統計）

## 執行結果

- 排除非特徵中繼欄 `date_decision`，分析 **194 個候選特徵**（172 數值、22 類別）。
- 以 DuckDB reservoir sampling（固定 seed=42）取出 **500,000 列**（所有欄位共用同一批列），
  全部以與 `feature_selection_woe_iv.py` 相同的決策樹分箱 / 類別稀有併箱 +
  `category_encoders.WOEEncoder` 轉換為 WOE 值，組成 500,000 × 194 WOE 矩陣。
- 總執行時間：287 秒（約 5 分鐘），全程無錯誤。
- VIF 範圍：**1.008 ~ 155,029.8**（中位數 11.20）。
- 分層統計：

  | 層級 | 欄位數 | 佔比 |
  |---|---|---|
  | 高（建議剔除，VIF>10） | 104 | 53.6% |
  | 中（留意，5≤VIF≤10） | 32 | 16.5% |
  | 低（可接受，VIF<5） | 58 | 29.9% |

- **超過一半（104/194）的候選特徵存在高度多重共線性**，反映出 9 種選法各自獨立挑選,
  對同一份底層資料的不同聚合窗口 / 版本重複選中的現象。本報告為 **report-only**,
  不自動剔除任何欄位,實際去留由使用者後續人工決定。

## 驗證（statsmodels 交叉比對，確認相關矩陣法與迴歸法一致）

挑選涵蓋整個 VIF 量級範圍（1.01 ~ 155,030）的 5 個欄位，另以
`statsmodels.stats.outliers_influence.variance_inflation_factor`（逐欄 OLS 回歸,
加入常數項)獨立重算，與相關矩陣虛擬逆法比對：

| column | corr-inverse VIF | statsmodels VIF |
|---|---|---|
| `pmtamount_36A_positive_count` | 155029.8311 | 155029.8310 |
| `pmts_dpd_303P_mean__max` | 39.8531 | 39.8531 |
| `pmts_overdue_1152A_std__mean` | 11.2548 | 11.2548 |
| `maxdpdlast3m_392P` | 4.3469 | 4.3469 |
| `maxannuity_4075009A` | 1.0082 | 1.0082 |

兩法在跨越 5 個數量級的 VIF 值上**皆一致到小數點後 4 位**，確認相關矩陣虛擬逆法
（快速、記憶體輕量）與 statsmodels 的標準回歸定義數學等價、實作正確。

## Top 10 最高 VIF 欄位

| column | source | var_type | VIF |
|---|---|---|---|
| `pmtamount_36A_positive_count` | TOP150入選-2 | numeric | 155,029.83 |
| `pmtamount_36A_non_null_count` | Catboost | numeric | 38,756.75 |
| `tax_registry_c_count` | TOP150入選-2 | numeric | 38,756.75 |
| `amount_4527230A_positive_count` | Catboost | numeric | 27,493.05 |
| `tax_registry_a_count` | TOP150入選-2 | numeric | 27,487.76 |
| `relationshiptoclient_415T_mode` | TOP150入選-2 | categorical | 2,036.997 |
| `relationshiptoclient_642T_mode` | TOP150入選-2 | categorical | 2,036.920 |
| `b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed` | AUC+KS | numeric | 888.20 |
| `b2_pmts_dpdvalue_108P_overdue_rate_recomputed` | AUC+KS | numeric | 887.58 |
| `days360_512L` | TOP150入選-2 | numeric | 158.62 |

## 合理性抽驗（確認分析確實捕捉到已知的共線性來源）

1. **`pmtamount_36A_positive_count` vs `pmtamount_36A_non_null_count`**：同一原始欄位的
   「正值筆數」與「非空筆數」統計量，本質上高度重疊 → 兩者及與其他欄位組合後的
   VIF 皆落在數萬等級，符合預期。
2. **`relationshiptoclient_415T_mode` vs `relationshiptoclient_642T_mode`**：這兩欄在
   `person_join_cleaning_aggregation_plan.md` 中已記錄「關係人列中 358,128 筆同時有值,
   其中僅 70.6% 相同,非完全冗餘但高度相關」，此次 VIF 分析（皆 ≈2,037）與該既有發現
   完全吻合，確認方法正確地偵測出已知的重複資訊來源。
3. **`pmts_dpd_303P_*` 同源不同視窗家族**（13 個變體：`mean`、`recent12_mean`、
   `recent6_mean`、`recent3_mean`、`std`、`overdue_rate` 等）：呈現清楚的梯度 ——
   衡量「逾期程度」的核心統計量（mean/std/overdue_rate 系列）VIF 落在 12~47（高共線）,
   而結構性計數欄（`non_null_count`、`n_unique`、`trend`、`recent_time_key`）VIF 僅
   1.2~4.9（低共線）,顯示不同滾動視窗的中心趨勢統計量彼此高度冗餘,但計數/結構類
   指標仍保有獨立資訊,符合特徵工程的直覺預期。
4. **我方 WOE 篩選的 10 個特徵中,有 8 個（80%）落在高 VIF 層**——顯示先前 IV 篩選
   （單變量檢定)挑出的強特徵彼此之間也存在明顯共線性,IV 高不代表彼此獨立,
   後續整合特徵集合時需一併納入 VIF 結果考量。

## 已知限制

- VIF 基於 500,000 列的固定隨機樣本（非全量 150 萬列),在記憶體有限(約 5GB 可用)下
  的折衷選擇;VIF 統計量在此樣本數下已具統計穩定性,但極端情況的精確值可能與全量
  計算有微小差異。
- VIF 計算對象是 **WOE 轉換後的數值**,而非原始欄位值——這與「若採 WOE 評分卡」的
  實際建模輸入一致,但若後續改用其他編碼方式（如 one-hot、raw numeric),VIF 結果
  需重新計算。
- 決策樹分箱含隨機抽樣(見 `feature_selection_woe_iv.py` 的已知限制),個別欄位的
  WOE 值/VIF 在不同執行間可能有小幅波動,但分層結果應穩定。
- 本報告為 report-only,未執行自動迭代剔除(如「每次移除最高 VIF 欄位直到全部
  ≤ 門檻」的 backward elimination);194 欄中 104 欄落在「高」層,若採嚴格 VIF<10
  門檻剔除,需搭配業務/IV 權重人工決定保留哪個代表欄位,而非全數移除。
