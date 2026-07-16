# applprev_1 15 欄位清洗與聚合 — 完成摘要

## 概述

依 `applprev_1_join_cleaning_aggregation_plan.md` 第一階段計畫實作 [data_preprocessing_2.py](data_preprocessing_2.py),對 `train_applprev_join.parquet`(6,525,979 列、1,221,522 個 case_id)中 15 個指定欄位做清洗、缺失值處理,並以 `case_id` 為 key 聚合成 case 層級特徵表。

**輸出**:`data/train_applprev_1_agg.parquet`,1,221,522 列(= distinct case_id)× 57 欄(本階段)。

## 欄位類型判定(feature_definitions.csv + DuckDB 驗證)

| 欄位 | 型別 | 缺失率 | 類型 | 說明 |
|---|---|---|---|---|
| actualdpd_943P | DOUBLE | 0.04% | 連續型 | 實際逾期天數;p99=0、max=4206 |
| byoccupationinc_3656910L | DOUBLE | 76.5% | 連續型 | 先前申請職業收入 |
| cancelreason_3545846M | VARCHAR | 0%* | 離散型 | 申請取消原因 |
| credacc_credlmt_575A | DOUBLE | 3.0% | 連續型 | 信用卡額度 |
| currdebt_94A | DOUBLE | 34.4% | 連續型 | 目前債務 |
| education_1138M | VARCHAR | 0%* | 離散型 | 教育程度 |
| employedfrom_700D | 日期字串 | 59.6% | 日期型 | 就業起始日 |
| mainoccupationinc_437A | DOUBLE | 1.6% | 連續型 | 主要收入金額 |
| maxdpdtolerance_577P | DOUBLE | 47.4% | 連續型 | 最大容忍逾期天數 |
| outstandingdebt_522A | DOUBLE | 34.6% | 連續型 | 未償債務 |
| profession_152M | VARCHAR | 0%* | 離散型 | 職業(11,508 類,98.9% 佔位值) |
| rejectreason_755M | VARCHAR | 0%* | 離散型 | 拒絕原因 |
| rejectreasonclient_4145042M | VARCHAR | 0%* | 離散型 | 客戶端拒絕原因 |
| revolvingaccount_394A | DOUBLE | 95.5% | **帳戶編號**(非金額) | 循環帳戶 |
| status_219L | VARCHAR | 0.001% | 離散型 | 先前申請狀態 |

\* M 欄無 NULL,但 `'a55475b1'` 為雜湊「缺失/無」佔位類別。

## 清洗規則

1. **99%PR 截尾(winsorize,不刪列)**:6 個連續欄(`byoccupationinc_3656910L`、`credacc_credlmt_575A`、`currdebt_94A`、`mainoccupationinc_437A`、`maxdpdtolerance_577P`、`outstandingdebt_522A`)超過全體 99 分位者以 p99 取代。
   - **不套用 `actualdpd_943P`**:p99=0,截尾會抹掉全部逾期訊號(使用者決議)。
   - **不套用 `revolvingaccount_394A`**:帳戶編號,改為二元旗標。
2. **A 結尾負值清除**:5 個 A 欄負值設 NULL(實測無負值,作為防護)。
3. **D 結尾未來日處理**:`employedfrom_700D` 以該筆申請 `creationdate_885D` 為基準,就業起始日晚於申請日(56 列)設 NULL,保留該列其他欄位。
4. **就業年資衍生欄**:`tenure_years = 2024 - (year + (month-1)/12)`。

## 缺失值處理

| 欄位 | 方式 | 理由 |
|---|---|---|
| actualdpd_943P | 補 0 | DPD 缺失 = 無逾期紀錄,語意等同 0 |
| maxdpdtolerance_577P / currdebt_94A / outstandingdebt_522A / credacc_credlmt_575A | 補 0 | 缺失集中於 D/T(被拒/取消,無合約),結構性缺失補 0 |
| mainoccupationinc_437A | 不補值 | 收入不可能為 0;聚合統計天然忽略 null,補值反注入假值 |
| byoccupationinc_3656910L | 不補值 + `null_ratio` 特徵 | 缺失 76.5%,缺失本身可能有訊號 |
| employedfrom_700D / tenure_years | 不補值 | 日期無合理填補值 |
| revolvingaccount_394A | 轉二元 `has_revolving`,null→0 | 帳戶編號,數值統計無意義 |
| 5 個 M 類別欄 | 不補;`'a55475b1'` 保留為獨立類別 | 佔位值即「無/未填」類別,對統計有意義 |
| status_219L | 66 筆 NULL 不補,比率計算排除於分母 | 佔比 0.001%,補值是雜訊 |

## 聚合特徵(group by case_id,聚合前按 creationdate 排序使 last=最近一次)

### 連續型
| 欄位 | 指標 | 理由 |
|---|---|---|
| actualdpd_943P | max、mean、n_events(>0 次數) | max 抓最嚴重逾期;此欄 99% 為 0,逾期「次數」比金額分布更穩健 |
| maxdpdtolerance_577P | max、mean | max 為歷史最差表現 |
| currdebt_94A | sum、max、last | sum=總負債水位;last=最近債務現況 |
| outstandingdebt_522A | sum、max、last | 未償債務總額對應償債壓力 |
| credacc_credlmt_575A | max、last | 額度反映授信評價;sum 無意義(非流量) |
| mainoccupationinc_437A | last、max、mean | last 最重要(最近申報最接近現況) |
| byoccupationinc_3656910L | last、mean、null_ratio | 缺失率 76.5%,null_ratio 本身作為特徵 |

### 日期型
| 欄位 | 指標 | 理由 |
|---|---|---|
| tenure_years | sum(使用者指定)、last、max | sum 與申請次數共線,故建議以 last(最近年資,現況)、max(就業穩定度)為主力 |

### 離散型(5 個 M 欄)
每欄輸出 `mode`、`mode_ratio`、`n_unique`、`entropy`、`non_placeholder_ratio`;`education_1138M` 另取 `last`。profession(98.9% 佔位值)mode 無鑑別力,以 n_unique 與非佔位值比例為主要訊號。

### status_219L 與拒絕率特徵(使用者指定需求)
已驗證 `D` = 被拒絕(0% approvaldate、61% 有 rejectreason)。

```
reject_rate = count(status_219L == 'D') / count(status_219L is not null)   # 依 case_id 分組
```

配套特徵:`n_prev_apps`(先前申請總數)、`n_rejected`、`n_status_valid`、`last_status`(最近一次狀態)、`cancel_rate`(status=='T' 取消率)。

## 驗證結果(DuckDB 抽驗)

| 檢查項 | 結果 |
|---|---|
| 輸出列數 | 1,221,522 = distinct case_id ✓ |
| reject_rate | ∈ [0,1];3 個 case_id 手工比對 D 件數/總件數皆正確 ✓ |
| 截尾 | `currdebt_94A_max` 上限 ≈ p99;`actualdpd_943P_max` 仍為 4206(未截尾)✓ |
| tenure_years | 無負值(未來日已清除)✓ |
| has_revolving | 僅 {0,1} ✓ |
| 補 0 欄位 | 聚合後無 NULL ✓ |
