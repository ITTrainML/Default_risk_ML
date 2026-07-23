# Train/Test 建模特徵管線重建 — 完成摘要

> **v2 更新(資料工程/特徵工程分區 + 財務指標)**：見文末「## v2 擴充」章節。以下概述已更新為 v2 狀態。

## 概述

依 [build_model_dataset_plan.md](build_model_dataset_plan.md) 計畫實作
[build_model_dataset.py](build_model_dataset.py)，從 `data/train/`、`data/test/` 的原始分區檔
重建最終建模特徵,輸出 `df_train`、`df_test`(Polars DataFrame + parquet)。程式以
**【資料工程 DATA ENGINEERING】**(union、標籤型別標準化、日期轉 _days)與
**【特徵工程 FEATURE ENGINEERING】**(清洗、聚合、財務指標、衍生欄、join)兩大區塊組織。

**輸出(v2)**：
- `data/df_train.parquet`：1,526,659 列 × **98 欄**(case_id、WEEK_NUM、date_decision、target + 94 特徵)
- `data/df_test.parquet`：10 列 × **97 欄**(無 target)
  （Kaggle 本地 test 樣本本身僅 10 個 case_id,屬正常現象——正式測試集在 Kaggle 端）

## 關鍵技術決策與其後果

1. **credit_bureau_a_2 改用 DuckDB(而非 Polars)**：此表訓練集達 **1.88 億列**（758 萬個
   `(case_id, num_group1)` 群組）。回收的 `bureau_a2_merge_GroupRule.txt` 本就指名
   `regr_slope`(SQL 標準函式)計算 trend，確認原設計即為 SQL 取向。用 DuckDB 的
   `regr_slope`(trend)、gaps-and-islands 視窗函式(longest_good_streak)、`FILTER`/`NULLIF`
   等原生聚合,比在 Polars 中重建同等邏輯更穩健。
2. **記憶體調校(過程中歷經 3 次 OOM 才穩定)**：
   - 第 1 次:預設 `memory_limit`(未設)在 2.7GB 用盡。
   - 第 2 次:提高到 6GB、加 `preserve_insertion_order=false`,仍在 5.5GB 用盡——因整條
     清洗→分箱→streak→L1 聚合鏈是**多層 TEMP VIEW 疊代**,DuckDB 會嘗試融合成一個巨大查詢計畫。
   - **解法**:把每個重運算階段(清洗+時間鍵、streak、L1 聚合)個別 `COPY ... TO parquet`
     寫成獨立 checkpoint 檔,下一階段用 `read_parquet()` 重新讀入,強迫 DuckDB 分階段執行、
     bound 各階段峰值記憶體。
   - 第 3 次:仍在 7.4GB 用盡,定位到 3 個 `COUNT(DISTINCT ...)`(逐群組維護精確去重雜湊集,
     對 758 萬群組開銷極大)。**改用 `approx_count_distinct`**(HyperLogLog)——因每組內基數
     天生很小(一份合約最多幾十筆繳款紀錄),近似值與精確值實質相同,但記憶體開銷大幅降低。
   - 最終以 `memory_limit='8GB'` + 上述兩項優化,一次執行成功(train+test 合計數分鐘內完成)。
3. **`regr_slope` 對樣本不足/x 無變異的群組回傳 `NaN`(非 SQL NULL)**:統一以
   `fill_nan(None)` 轉為 null,避免下游誤把 NaN 當成有效數值。
4. **`SUM`/`COUNT(DISTINCT)` 在 DuckDB 中會提升為 `HUGEINT`,轉 Polars 後變成
   `Decimal(38,0)`**:對這類欄位顯式 `::BIGINT` 轉型,確保輸出欄位型別乾淨一致
   （與其餘 Int64/Float64 欄位一致，避免下游 ML 套件對 Decimal 型別的相容性問題）。
5. **`approx_quantile` 對高度右偏/近全零欄位可能回傳極小浮點雜訊值(如 `2.2e-308`)而非精確
   `0.0`**：winsorize 的「若 p99==0 則不截尾」判斷改用容忍值 `abs(cap) < 1e-9`，避免這類雜訊
   誤觸發截尾、把所有正值誤壓成趨近於 0。

## 驗證結果

| 檢查項 | 結果 |
|---|---|
| `df_train` 列數 = `train_base` distinct case_id 數 | 1,526,659 = 1,526,659 ✓ |
| `df_test` 列數 = `test_base` distinct case_id 數 | 10 = 10 ✓ |
| 欄位集合一致性 | `train − test = {target}`；`test − train = {}` ✓ |
| 重複 case_id | train/test 皆 0 ✓ |
| `reject_rate` 值域 | [0.0, 1.0] ✓ |
| `age_years_appl` 值域 | 20.96 ~ 76.04（與既有 `person_join_cleaning_aggregation_plan.md` 實際執行結果**完全一致**）✓ |
| `target` 分布 | 0: 1,478,665、1: 47,994（與既有 `base_final.parquet` 完全一致）✓ |
| A 結尾欄位負值 | 全數欄位 min ≥ 0 ✓ |
| D 結尾欄位未來日 | 最大值 2020-10-05(與已知 `date_decision` 上界一致)✓ |
| winsorize 截尾生效 | `totalamount_996A__mean` 聚合後最大值 2,466,908.5 **精確等於**原始欄位 p99 截尾值,原始未截尾 max 高達 13.9 億 ✓ |
| depth-2 二次聚合正確性(獨立手算比對) | `pmts_dpd_303P_mean_positive__recomputed`:test case 57569 → 2328.571429(手算)= 2328.571429(程式)✓;train case 7148 → 510.7(手算)= 510.7(程式)✓ |
| depth-2 std / streak 正確性 | `pmts_dpd_303P_std__max`(case 57569)57.806739 手算=程式 ✓；`longest_good_streak`(case 57631)5 段連續 0 值手算=程式 ✓ |

## 已知限制與後續建議

- **approx_count_distinct** 取代精確 `COUNT(DISTINCT)`:在本資料集(每組基數天生 ≤ 數十)下
  與精確值實質相同,但屬近似演算法,極端情況可能有 ±1 誤差。
- **credit_bureau_b(b1/b2)全數忽略**:依使用者指示捨棄,故原候選清單中的 3 個
  `b2_pmts_dpdvalue_108P_*` 欄位未包含在最終輸出中。
- **year/month 欄位配對的 active/closed 標籤**:Kaggle 官方 `feature_definitions.csv` 對
  `pmts_month_158T`/`706T` 的 active/closed 描述與對應 `pmts_year_1139T`/`507T` 的描述互相矛盾
  （已知的資料集文件錯誤）。本實作依「year 欄描述自洽、與同側 DPD 欄位語意一致」為準：
  `(pmts_year_1139T, pmts_month_158T)` 配對 `pmts_dpd_1073P`(active）、
  `(pmts_year_507T, pmts_month_706T)` 配對 `pmts_dpd_303P`(closed），與 build_model_dataset_plan.md
  已載明的假設一致。
- **`mean_fallback`/`max_fallback` 二次聚合定義**:`feature_name_mapping.csv` 僅明確定義
  `recomputed`/`weighted_avg`/`duration`，`fallback` 系列依欄位命名慣例實作為「一次聚合值
  以 0 補值後取 mean/max」，為本次實作中唯一依命名慣例推斷（而非文件明載）的計算方式。
- 中繼 checkpoint 檔(`.tmp/a2_keyed_*.parquet` 等)於每次執行後自動清除，不會殘留佔用磁碟。

---

## v2 擴充：資料工程/特徵工程分區 + 財務指標

在 v1(84 特徵)基礎上擴充,新增 10 欄(94 特徵),並將程式重構為兩大工程區塊。

### 新增/變更內容

1. **【資料工程】/【特徵工程】區塊分離**:程式以 `#####` banner 分隔兩區,各 `build_*`
   函式內以 `# [資料工程]` / `# [特徵工程]` 標註每一步。
2. **標籤型別標準化 `transform_by_label`**(資料工程):D→Date、A/P→Float64、M→Utf8、
   T/L→維持原生型別。驗證:`*M` 欄輸出為 String、`*A`/`*P` 為 Float64、`date_decision` 為 Date。
3. **日期差 `_days`**(資料工程):`date_decision − {lastrejectdate_50D, maxdpdinstldate_3546855D,
   lastdelinqdate_224D}` → `_days`;原 3 個 D 欄於特徵工程階段移除。
4. **新清洗規則**(特徵工程):A 欄負值→null、`_days` 欄負值→null(負 days=未來日,不合理)、
   D 欄移除。驗證:3 個 `_days` 欄 min=0(無負值)、max 合理(≤ 4666 天 ≈ 12.8 年)。
5. **7 個財務基礎欄聚合**(輸出用原始欄名):

   | 欄位 | 來源 | 統計 |
   |---|---|---|
   | totaldebt_9A、maininc_215A | static_0 | 直取(1:1) |
   | mainoccupationinc_384A | person_1 | max(本人列取回) |
   | totaldebtoverduevalue_178A、totaloutstanddebtvalue_39A | credit_bureau_a_1 | sum(跨合約) |
   | amtdepositincoming_4809444A、amtdebitoutgoing_4809440A | other_1(新 builder) | sum(1:1) |

6. **3 個財務比率特徵**(依 `Financial indicators.csv`):
   - `dti_ratio` = totaldebt_9A / coalesce(maininc_215A, mainoccupationinc_384A) — 償債能力
   - `overdue_debt_ratio` = totaldebtoverduevalue_178A / totaloutstanddebtvalue_39A — 逾期風險
   - `deposit_to_debt_ratio` = amtdepositincoming_4809444A / amtdebitoutgoing_4809440A — 流動性

   除零/除 null 以 `_safe_ratio` 保護回傳 null。
7. **`date_decision` 保留於輸出**(本次清單已列;v1 曾移除)。

### v2 驗證結果

| 檢查項 | 結果 |
|---|---|
| 列數 | train 1,526,659 / test 10;欄集合 train−test={target} ✓ |
| 欄數 | train 98 / test 97(v1 87/86 + 11 新欄) ✓ |
| 原 3 個 D 欄移除、`_days` 存在 | ✓ |
| `_days` 無負值 | 3 欄 min 皆 0 ✓ |
| 財務比率 ≥ 0、無 Decimal 欄 | ✓ |
| 標籤 Transform 型別 | M=String、A/P=Float64、date_decision=Date ✓ |
| **財務欄/比率手算比對** | case 491:sum(178A)=16.614、sum(39A)=55323.938、ratio=0.0003003 手算=程式 ✓;case 562:100.984/280177.8=0.00036 ✓;`mainoccupationinc_384A` max(43000/70000)手算=程式 ✓;DTI coalesce 備援(maininc 缺失時改用職業收入)驗證正確 ✓ |

### v2 已知限制

- **財務比率極端值**:`dti_ratio`(max≈12,762)、`overdue_debt_ratio`(max≈4.73M)因分母極小
  而出現極大值。本階段**未對衍生比率再截尾**(截尾僅套用於聚合前的連續基礎欄,符合需求 5);
  後續特徵工程若需要,可對這 3 個比率另做 winsorize/對數轉換。
- **`deposit_to_debt_ratio` 缺失率高(98.4%)**:`other_1` 僅覆蓋約 3.3% 的 case(51,109/1,526,659),
  其餘 case 無存款/支出紀錄故為 null;屬結構性缺失,非錯誤。
- **`overdue_debt_ratio` 缺失率 26.5%**:credit_bureau_a_1 中 178A/39A 約 91.8% 列為 null
  (僅存續合約有值),case 層級 sum 後仍有相當比例 case 完全無存續合約紀錄。
