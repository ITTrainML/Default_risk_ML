# train_person_1.parquet 清洗與聚合規劃

## 資料來源

- **來源檔案**: `DATA/train_person_1.parquet`
- **聚合參考**: `DATA/train_person_2_aggregated.parquet`
- **欄位定義**: `DATA/feature_definitions.csv`

---

## 一、處理欄位清單（25 個欄位）

| 欄位名稱 | 類型 | 原始 dtype | Null Rate | 說明 |
|---------|------|-----------|-----------|------|
| birth_259D | 日期型 (D) | String | 48.7% | 出生日期 |
| birthdate_87D | 日期型 (D) | String | 99.2% | 出生日期 |
| childnum_185L | 離散整數型 (L) | Float64 | 99.7% | 子女數 |
| contaddr_district_15M | 類別型 (M) | String | 0.0% | 聯絡人地址區域 |
| education_927M | 類別型 (M) | String | 0.0% | 教育程度 |
| empl_employedfrom_271D | 日期型 (D) | String | 80.9% | 受僱起始日 |
| empl_employedtotal_800L | 離散整數型 (L) | String | 82.2% | 受僱總長度 |
| empl_industry_691L | 類別型 (L) | String | 82.4% | 受僱行業 |
| familystate_447L | 類別型 (L) | String | 75.5% | 家庭狀態 |
| housetype_905L | 類別型 (L) | String | 96.6% | 房屋類型 |
| housingtype_772L | 類別型 (L) | String | 99.7% | 住宅類型 |
| incometype_1044T | 類別型 (T) | String | 48.7% | 收入類型 |
| mainoccupationinc_384A | 連續型 (A) | Float64 | 48.7% | 主要職業收入 |
| maritalst_703L | 類別型 (L) | String | 99.6% | 婚姻狀態 |
| persontype_1072L | 離散整數型 (L) | Float64 | 0.2% | 人員類型 |
| persontype_792L | 離散整數型 (L) | Float64 | 21.6% | 人員類型 |
| registaddr_district_1083M | 類別型 (M) | String | 0.0% | 註冊地址區域 |
| registaddr_zipcode_184M | 類別型 (M) | String | 0.0% | 註冊地址郵遞區號 |
| relationshiptoclient_415T | 類別型 (T) | String | 72.9% | 與客戶關係 |
| relationshiptoclient_642T | 類別型 (T) | String | 72.9% | 與客戶關係 |
| remitter_829L | 布林型 (L) | Boolean | 72.9% | 是否為匯款人 |
| role_1084L | 類別型 (L) | String | 0.2% | 角色類型 |
| safeguarantyflag_411L | 布林型 (L) | Boolean | 48.7% | 擔保標記 |
| type_25L | 類別型 (L) | String | 0.2% | 聯絡類型 |

---

## 二、資料清洗規則

### 規則 2：連續型特徵離群值（Cap 99%）

**適用欄位**：
| 欄位 | 處理邏輯 |
|------|---------|
| `mainoccupationinc_384A` (Float64) | 計算 P99，若 P99 > 0，則 > P99 者設為 P99；若 P99 = 0 則不處理 |
| `childnum_185L` (Float64) | 同上邏輯 |

**實作方式**：
```
p99 = mainoccupationinc_384A.quantile(0.99)
if p99 > 0:
    mainoccupationinc_384A = min(mainoccupationinc_384A, p99)
```

### 規則 3：含 A 欄位負值取代為 null

**適用欄位**：
| 欄位 | 處理邏輯 |
|------|---------|
| `mainoccupationinc_384A` | 若 < 0 → 設為 null |

### 規則 4：日期欄位未來日處理

**檢查邏輯**：若日期 > 今天 → 不合理（出生日不可能在未來），設為 null

**適用欄位**：
| 欄位 | 合理性檢查 |
|------|-----------|
| `birth_259D` | 出生日期在未來 → null |
| `birthdate_87D` | 出生日期在未來 → null |
| `empl_employedfrom_271D` | 受僱起始日在未來 → null |

### 其他欄位（類別型/布林型）

**無需清洗**，保留原始值（含 null），後續由聚合函數處理。

---

## 三、聚合流程

### Step 1：清洗 → `DATA/train_person_1_cleaned.parquet`

- 程式碼：`train_person_1_cleaned.py`
- 輸出：清洗後的 train_person_1 資料

### Step 2：JOIN → `DATA/train_person_2_join_1.parquet`

- 程式碼：`train_person_2_join_1.py`
- **左表**：`train_person_1_cleaned.parquet`（37 欄, ~2,973,991 rows）
- **右表**：`train_person_2_aggregated.parquet`（58 欄, 1,561,280 rows）
- **JOIN KEY**：`[case_id, num_group1]`
- **JOIN 方式**：LEFT JOIN（保留 person_1 所有資料，無匹配時 person_2 欄位為 null）
- **結果欄位數**：37 + 58 - 2(重複key) = **93 欄**

### Step 3：以 case_id 聚合 → `DATA/train_person_agg.parquet`

- 程式碼：`train_person_agg.py`
- **聚合 KEY**：`case_id`（一個 case_id 可能對應多個 num_group1）

---

## 四、train_person_1 原始欄位二次聚合統計指標

### 4.1 連續型數值特徵（mainoccupationinc_384A）

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 反映個人平均收入水準 |
| **std** | 反映同一 case_id 下多人收入的差異程度 |
| **min / max** | 捕捉極端收入情況 |
| **median** | 不受離群值干擾，更能代表中心趨勢 |
| **sum** | 計算家庭總收入 |
| **null_count** | 反映收入資料缺失程度 |

### 4.2 類別型特徵（education_927M, housetype_905L 等）

| 建議指標 | 原因說明 |
|---------|---------|
| **mode** | 該 case_id 下最常見的類別，代表主要特徵 |
| **mode_ratio** | 眾數占比，數值越高代表該 case 內一致性越高 |
| **n_unique** | 反映該 case 內多人的類別多樣性（家庭成員教育程度是否多元） |
| **entropy** | 資訊熵，量化類別分布的不確定性/混亂程度 |
| **null_count / null_rate** | 反映資料缺失程度 |

### 4.3 日期型特徵（birth_259D, empl_employedfrom_271D）

| 建議指標 | 原因說明 |
|---------|---------|
| **min_date / max_date** | 計算最早和最晚日期，可推算年齡範圍 |
| **date_range_days** | 日期跨度（最大-最小天數），反映多人日期分散程度 |
| **recency_days** | 距今天數，可轉換為年齡（如 min(birth) 可算出最大年齡） |
| **null_count** | 反映缺失程度 |

### 4.4 離散計數型（childnum_185L, empl_employedtotal_800L, persontype_1072L）

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 平均子女數/平均受僱年資 |
| **max** | 最大子女數/最長年資（可能為主要決策者） |
| **sum** | 總和（如家庭總子女數） |
| **n_unique** | 同一 case 內不同 persontype 的多樣性 |
| **null_count** | 反映缺失程度 |

### 4.5 布林型特徵（remitter_829L, safeguarantyflag_411L）

| 建議指標 | 原因說明 |
|---------|---------|
| **mode** | 該 case 最主要的布林值 |
| **mode_ratio** | True/False 的一致性比例 |
| **sum** | 計算 True 的個數（如該 case 有多少人是 remitter） |
| **null_count** | 反映缺失程度 |

---

## 五、train_person_2_aggregated 二次聚合統計指標分析

`train_person_2_aggregated` 已是以 `[case_id, num_group1]` 聚合過的資料，包含 8 個特徵各 7 個指標：

| 已聚合的指標 | 資料類型 | 數值範圍 |
|------------|---------|---------|
| `_mode` | String | 類別值 |
| `_mode_ratio` | Float64 | [0, 1] |
| `_n_unique` | UInt32 | 非負整數 |
| `_entropy` | Float64 | [0, ∞) |
| `_null_count` | UInt32 | 非負整數 |
| `_null_rate` | Float64 | [0, 1] |
| `_non_null_count` | UInt32 | 非負整數 |

### 5.1 `_mode`（String 類別型）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **mode** | 同一 case_id 下各 person 中最常見的類別 |
| **n_unique** | 同一 case 中不同 person 有多少種不同 mode 類別 |
| **null_count** | 缺失數量 |

### 5.2 `_mode_ratio`（Float64, [0,1]）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 各 person 眾數占比的平均值，反映整體信心度 |
| **min** | 最低的眾數占比，可能有某一 person 類別分布較分散 |
| **std** | 各 person 眾數占比的變異程度 |

### 5.3 `_n_unique`（UInt32）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 平均每個 person 的唯一值數量 |
| **max** | 最大唯一值數量，可能反映特殊情況 |
| **sum** | 加總所有 person 的唯一值數量 |

### 5.4 `_entropy`（Float64）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 各 person 資訊熵的平均值，反映平均混亂程度 |
| **max** | 最高混亂程度的 person |
| **std** | 各 person 間熵值的差異 |

### 5.5 `_null_count`（UInt32）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **sum** | 該 case 下所有 person 的缺失總數 |
| **mean** | 平均每個 person 的缺失數量 |

### 5.6 `_null_rate`（Float64, [0,1]）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **mean** | 各 person 缺失率的平均值 |
| **max** | 缺失率最高的 person，可能資料品質較差 |

### 5.7 `_non_null_count`（UInt32）二次聚合

| 建議指標 | 原因說明 |
|---------|---------|
| **sum** | 該 case 下所有 person 的非缺失總數 |
| **mean** | 平均每個 person 的非缺失數量 |

---

## 六、資料流摘要

```
train_person_1.parquet (2,973,991 rows, 37 cols)
        │
        ▼  [清洗：離群值cap、負值轉null、未來日處理]
train_person_1_cleaned.parquet
        │
        ▼  [LEFT JOIN on case_id + num_group1]
        │  左表：train_person_1_cleaned
        │  右表：train_person_2_aggregated (1,561,280 rows, 58 cols)
        ▼
train_person_2_join_1.parquet (~2.97M rows, 93 cols)
        │
        ▼  [GROUP BY case_id 二次聚合]
        │  - raw 欄位用對應指標
        │  - 已聚合欄位用二次指標
        ▼
train_person_agg.parquet (依 case_id 唯一值 rows)