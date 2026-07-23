# build_model_dataset.py 擴充計畫：資料工程/特徵工程分區 + 財務指標

## Context

延續已完成並驗證的 [build_model_dataset.py](build_model_dataset.py)
（從 `data/train/`、`data/test/` 原始分區重建 84 個建模特徵 → `df_train`/`df_test`）。
本次要**擴充並重構**該管線,新增六項需求:

1. **程式與 plan 明確區分「資料工程(DATA ENGINEERING)」與「特徵工程(FEATURE ENGINEERING)」兩區塊**,
   程式加上區塊註解(需求 2)。
2. **依欄位標籤做格式標準化 Transform**(資料工程,需求 3):P=DPD 天數→數值、M=遮罩類別→字串類別、
   A=金額→數值、D=日期→Date、T/L=未指定(維持原生型別)。
3. **`date_decision` 減 3 個日期欄 → `_days` 欄**(資料工程,需求 4):`lastrejectdate_50D`、
   `maxdpdinstldate_3546855D`、`lastdelinqdate_224D` → `{col}_days`,原 D 欄於特徵工程階段移除。
4. **新的清洗規則**(特徵工程,需求 5):連續型 99%PR 截尾(p99==0 不補)、A 欄負值→null、
   **`days` 欄負值→null**、**D 欄已轉 days 故移除**。
5. **7 個財務基礎欄依 `Financial indicators.csv` 選統計指標聚合**(需求 6),並在本 plan 說明選用理由。
6. **建構 3 個財務比率特徵**(需求 9),名稱與說明記於本 plan。

**已與使用者確認的決策**:
- **繼續忽略 credit_bureau_b(b1/b2)** → 維持捨棄 3 個 `b2_pmts_dpdvalue_108P_*` 特徵。
- **7 個財務基礎欄「聚合後」保留為輸出特徵**(在「只用下列欄位」清單中),同時作為財務比率的輸入。
- **DTI 分母用 coalesce**:`totaldebt_9A / coalesce(maininc_215A, mainoccupationinc_384A)`。
- 輸出 `df_train`/`df_test`(Polars)+ parquet;`date_decision` 本次**保留於輸出**(清單已列)。

## 資料工程 vs 特徵工程的界線(本次組織原則)

| 區塊 | 職責 | 對應函式 |
|---|---|---|
| **資料工程** | union 分區、依標籤標準化型別、日期轉 `_days`。「把原始資料整理成乾淨、正確型別的表」 | `read_table`、`transform_by_label`、`add_date_diff_days` |
| **特徵工程** | 合理值清洗(截尾/負值)、聚合(depth 0/1/2、後贅詞統計)、財務指標、5 衍生欄、join | `winsorize_p99`、`clean_by_label`、各 `build_*`、`build_financial_indicators` |

程式以區塊註解 banner 分隔,且各 `build_*` 函式內以 `# [資料工程]` / `# [特徵工程]` 標註每步。

## 來源表 / depth 對照(新增財務欄與 other_1)

| depth | 表 | 選中/財務欄 | 顆粒度 |
|---|---|---|---|
| 0 | static_0 | 既有 27 欄 + **totaldebt_9A、maininc_215A**(財務) | 1:1 |
| 0 | static_cb_0 | 既有 8 欄 | 1:1 |
| 1 | credit_bureau_a_1 | 既有 13 欄 + **totaldebtoverduevalue_178A、totaloutstanddebtvalue_39A**(財務) | 11.5 列/case |
| 1 | applprev_1 | 既有 4 欄 + reject_rate/last_status/tenure_years_max | 4.3 列/case |
| 1 | person_1 | 既有 5 `_appl` + age/tenure + **mainoccupationinc_384A**(財務) | 1.95 列/case |
| 1 | **other_1(新 builder)** | **amtdepositincoming_4809444A、amtdebitoutgoing_4809440A**(財務) | 1:1 |
| 2 | credit_bureau_a_2 | 既有 20 欄(DuckDB,不變) | 1.88 億列 |

## 【資料工程】步驟

### D1. union 分區檔（不變）
沿用 `read_table`:`{split}_{table}_{X}_*.parquet` union;單檔直讀。

### D2. 依標籤格式標準化 `transform_by_label(df, cols)`
逐欄依**大寫尾碼**標準化型別:

| 標籤 | 語意 | 標準化動作 |
|---|---|---|
| `*D` | Transform date | `str.strptime(pl.Date, strict=False)` |
| `*A` | Transform amount | `cast(pl.Float64)` |
| `*P` | Transform DPD(天數) | `cast(pl.Float64)` |
| `*M` | Masking categories | `cast(pl.Utf8)`(維持遮罩雜湊字串為類別) |
| `*T` / `*L` | Unspecified | 維持原生型別(不強制轉換) |

### D3. 日期差轉 `_days`  `add_date_diff_days(df, date_cols, ref="date_decision")`
對 `lastrejectdate_50D`、`maxdpdinstldate_3546855D`、`lastdelinqdate_224D`:
`{col}_days = (date_decision − col).dt.total_days()`。原 D 欄留待特徵工程 D4 移除。

## 【特徵工程】步驟

### F1. 合理值清洗（需求 5）
- **連續型 99%PR 截尾**(`winsorize_p99`,不刪列;`p99==0` 或近似 0 則不截尾)。
- **A 欄負值 → null**(`clean_by_label`,金額不可為負)。
- **`_days` 欄負值 → null**(負 days = 日期晚於 date_decision,不合理)。
- **D 欄移除**:`_days` 已取代,drop `lastrejectdate_50D` 等 3 個原始 D 欄。
- `_days` 欄視為連續型,亦套用 99%PR 截尾。

### F2. 聚合（depth 0/1/2,後贅詞決定統計指標；不變）
沿用既有 `stat_expr` / `NUMERIC_STAT_FUNCS` / `cat_group_stats` 與 DuckDB depth-2 邏輯。

### F3. 財務基礎欄聚合統計選用（需求 6 — 理由記錄）
7 欄依 `Financial indicators.csv` 之比率語意選聚合統計;**輸出沿用原始欄名**(代表 case 層級值):

| 財務欄 | 來源(depth) | 選用統計 | 選用理由 |
|---|---|---|---|
| `totaldebt_9A` | static_0(0) | 直取 | 已是 case 層級「總負債」,1:1 無需聚合 |
| `maininc_215A` | static_0(0) | 直取 | 已是 case 層級「主要收入」,1:1 無需聚合 |
| `mainoccupationinc_384A` | person_1(1) | **max** | 收入為申請人層級屬性,僅本人列(num_group1==0)有值、關係人列 100% 缺失;max 跳過 null 即取回本人收入,對任何雜訊非空值亦穩健 |
| `totaldebtoverduevalue_178A` | credit_bureau_a_1(1) | **sum** | 指標語意為「逾期債務佔總未償債務比例」;分子應為客戶所有存續合約的**逾期債務總額**,故跨合約加總 |
| `totaloutstanddebtvalue_39A` | credit_bureau_a_1(1) | **sum** | 同上,分母為跨合約的**未償債務總額**;sum 與分子同層級才能構成正確比率 |
| `amtdepositincoming_4809444A` | other_1(1) | **sum** | 流動性指標分子=存款流入總額;other_1 為 1:1,sum 即該值,且語意上總流入最能代表可用流動性 |
| `amtdebitoutgoing_4809440A` | other_1(1) | **sum** | 流動性指標分母=支出流出總額;sum 與分子同層級 |

### F4. 財務比率特徵（需求 9 — 名稱與說明記錄）
以 F3 聚合後(case 層級)欄位計算,除零/除 null 以 `NULLIF` 保護回傳 null:

| 特徵名 | 比率類別 | 計算 | 說明 |
|---|---|---|---|
| `dti_ratio` | 償債能力 | `totaldebt_9A / NULLIF(coalesce(maininc_215A, mainoccupationinc_384A), 0)` | Debt-to-Income:現行總負債相對收入之壓力;收入以主要收入為主、職業收入為備援(補 maininc 的 33% 缺失) |
| `overdue_debt_ratio` | 逾期風險 | `totaldebtoverduevalue_178A / NULLIF(totaloutstanddebtvalue_39A, 0)` | 存續合約中逾期債務佔總未償債務之比例,越高風險越大 |
| `deposit_to_debt_ratio` | 流動性 | `amtdepositincoming_4809444A / NULLIF(amtdebitoutgoing_4809440A, 0)` | 存款負債覆蓋率:存款流入相對支出流出,衡量以現有現金流償債之能力 |

### F5. 5 個衍生欄（需求 10;不變,公式已於前一版驗證）
`reject_rate`、`last_status`、`tenure_years_max`(applprev_1)、`age_years_appl`、`tenure_years_appl`
(person_1)——沿用既有 `build_applprev` / `build_person` 實作。

### F6. join 回 base（需求 8）
以 `{split}_base`(case_id、WEEK_NUM、date_decision、train 另含 target)為主表,left join:
static_0、static_cb_0、a_1、applprev_1、person_1、a_2、**other_1**、財務比率。
最後 `select` 最終欄位清單(見下)。

## 最終輸出欄位

鍵:`case_id`、`WEEK_NUM`、`date_decision`(+ train 的 `target`)。特徵 = 前一版 79 個非 b2 特徵,
其中 3 個 D 欄改為 `_days` 版本,**再加** 7 個財務基礎欄 + 3 個財務比率。b2 三欄維持排除。

## 產出檔案

- **改寫** `build_model_dataset.py`(區塊化 + 新增 `transform_by_label`/`add_date_diff_days`/
  `build_other`/`build_financial_indicators`;`build_static`/`build_bureau_a1`/`build_person` 擴充財務欄)。
- **覆寫** `data/df_train.parquet`、`data/df_test.parquet`。

## 驗證方式

1. 執行無錯誤;`df_train` 列數 = 1,526,659、`df_test` = test_base case 數;欄集合一致(test 少 target)。
2. `_days` 欄:無負值(已清 null)、最大值 ≈ 各自 p99;原 3 個 D 欄不在輸出。
3. 財務比率:`dti_ratio`/`overdue_debt_ratio`/`deposit_to_debt_ratio` 皆 ≥ 0;抽 2-3 個 case 手工
   比對(如 `overdue_debt_ratio` = sum(178A)/sum(39A))。
4. 財務基礎欄:`totaldebtoverduevalue_178A`(sum)、`mainoccupationinc_384A`(max)抽樣對照原始列。
5. 標籤 Transform:D 欄輸出前為 Date、A/P 為 Float64、M 為字串;抽查 dtype。
6. 沿用既有驗證(reject_rate∈[0,1]、age 合理、depth-2 手算比對)仍通過。
