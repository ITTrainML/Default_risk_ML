# train_person_1 清洗與聚合計畫

## Context

`train_person_1.parquet`(2,973,991 列,粒度 `(case_id, num_group1)`,1,526,659 個 case_id,
num_group1=0 為申請人本人、>0 為關係人/聯絡人,每個 case 皆有本人列)要與
`train_person_2_aggregated.parquet`(1,561,280 列,person_2 已聚合到 `(case_id, num_group1)`
層級,8 個類別來源欄 × 7 個指標,可完全 join 上 person_1 的鍵)合併,再以 `case_id` 為 key
聚合成 case 層級特徵表,供後續違約風險模型使用。

**產出**:
- `train_person_2_join_1.py`(DuckDB,風格沿用 `data_preprocessing_1.py`)→ `data/train_person_2_join_1.parquet`
- `train_person_agg.py`(Polars,風格沿用 `data_preprocessing_2.py`)→ `data/train_person_agg.parquet`
- 本文件:清洗規則、聚合設計與逐欄統計指標分析

## 探索結論(已用 DuckDB 驗證)

**24 個指定欄位類型判定**:

| 類型 | 欄位 |
|---|---|
| 連續型 | `childnum_185L`、`mainoccupationinc_384A` |
| 日期型 | `birth_259D`、`birthdate_87D`、`empl_employedfrom_271D` |
| 離散/布林型(其餘 19 欄) | `contaddr_district_15M`、`education_927M`、`empl_employedtotal_800L`、`empl_industry_691L`、`familystate_447L`、`housetype_905L`、`housingtype_772L`、`incometype_1044T`、`maritalst_703L`、`persontype_1072L`、`persontype_792L`、`registaddr_district_1083M`、`registaddr_zipcode_184M`、`relationshiptoclient_415T`、`relationshiptoclient_642T`、`remitter_829L`、`role_1084L`、`safeguarantyflag_411L`、`type_25L` |

**離群值/截尾**:`childnum_185L` p99=4、`mainoccupationinc_384A` p99=190,000,兩者 p99 皆 >0
→ 依規則都要截尾;`mainoccupationinc_384A`(A 欄)實測無負值,防護性實作負轉 null。

**未來日基準**:採 join `train_base.date_decision`(VARCHAR "YYYY-MM-DD",逐 case 不同,
範圍 2019-01-01~2020-10-05)。相對全域 max 檢查未發現未來日
(`birth_259D` max=1999-10-01、`empl_employedfrom_271D` max=2020-09-15),但規則仍以
「逐 case 相對 date_decision」實作,避免時間洩漏且與 applprev 階段基準一致。

**缺失結構完美二分(關鍵發現,決定聚合設計)**:

| 欄位 | 本人列(num_group1=0) | 關係人列(num_group1>0) |
|---|---|---|
| mainoccupationinc_384A 缺失率 | 0% | 100% |
| birth_259D 缺失率 | 0% | 100% |
| relationshiptoclient_415T 缺失率 | 100% | 44.4% |
| education_927M 佔位值比例 | 52.3% | 99.3% |

→ 本人與關係人在幾乎每個欄位上是互斥的資訊來源,**必須分流聚合**,否則對本人收入取
`mean`/`max` 等統計量沒有意義(本人恆只有一列)。

**其他發現**:
- `remitter_829L` 僅 `{False, null}`(n_unique=1)→ 死特徵,不產出模型欄,僅記錄於本文件。
- `birthdate_87D` 99.2% 全空,為 `birth_259D` 的冗餘/備援版本 → 僅作 `coalesce` 備援用於
  年齡衍生,不獨立產生統計欄。
- person_2_aggregated 中 92%(1,435,041/1,561,280)列屬本人(num_group1=0)。
- `relationshiptoclient_415T` 與 `642T` 在關係人列中 358,128 筆同時有值,其中僅 252,861
  筆(70.6%)相同 → 非完全冗餘,兩欄都保留但以 415T 為主指標。

## 步驟 1:join(train_person_2_join_1.py)

```sql
COPY (
  SELECT p1.*, p2.* EXCLUDE (case_id, num_group1), b.date_decision
  FROM 'data/train_person_1.parquet' p1
  LEFT JOIN 'data/train_person_2_aggregated.parquet' p2 USING (case_id, num_group1)
  LEFT JOIN 'data/train_base.parquet' b USING (case_id)
) TO 'data/train_person_2_join_1.parquet' (FORMAT PARQUET)
```

只取 `date_decision` 供日期衍生基準,不引入 base 的其他欄位。

## 步驟 2:清洗規則(train_person_agg.py,聚合前)

1. 只讀 join 檔的 24 指定欄 + `case_id`/`num_group1`/`date_decision` + 8 個 p2 來源欄的
   全部指標欄。
2. **連續欄 99%PR 截尾(不刪列)**:`childnum_185L`(cap=4)、`mainoccupationinc_384A`
   (cap=190,000);p99 以全體非空值現場計算,不寫死常數。
3. **A 欄負值 → null**:`mainoccupationinc_384A`(`col < 0` 轉 null,防護性實作)。
4. **D 欄未來日 → null**:`birth_259D`、`birthdate_87D`、`empl_employedfrom_271D` 晚於
   該列的 `date_decision` 者設為 null。
5. **日期衍生**:
   - `age_years = (date_decision − coalesce(birth_259D, birthdate_87D)) / 365.25`
   - `tenure_years = (date_decision − empl_employedfrom_271D) / 365.25`

## 步驟 3:聚合設計(group by case_id,本人/關係人分流)

### A. 本人特徵(`filter(num_group1==0)` 後直接取值,後綴 `_appl`)

| 欄位 | 類型 | 指標 | 理由 |
|---|---|---|---|
| age_years | 連續 | 直取 | 年齡為核心人口風險因子;本人僅一列,取 mean/max 無意義,直取即可 |
| tenure_years | 連續 | 直取 | 就業年資反映穩定度;已排除未來日 |
| mainoccupationinc_384A | 連續 | 直取(已截尾) | 當前所得為可負擔性核心變數 |
| childnum_185L | 連續 | 直取(已截尾) | **實作後驗證發現:此欄在本人列(num_group1=0)100% 缺失**(全 1,526,659 列皆 NULL),僅極少數關係人列有值(9,907/1,447,332≈0.68%)。原探索的 99.7% 缺失率是全表混合統計,未拆分本人/關係人,誤判其為「本人可用但接近死欄」。實際上 `childnum_185L_appl` 是**全空的死特徵**,建模前應直接捨棄;若需子女數訊號,應改向資料源頭(如 static 表)尋找替代欄位 |
| education_927M | 類別 | 直取 | 教育程度風險因子;佔位值 `a55475b1` 視為獨立類別保留 |
| empl_employedtotal_800L / empl_industry_691L | 類別 | 直取 | 任職年資分桶/產業別,反映就業穩定度 |
| incometype_1044T | 類別 | 直取 | 所得來源類型(受僱/退休等),社經風險因子 |
| familystate_447L / maritalst_703L | 類別 | 直取 | 婚姻家庭狀態;maritalst 99.6% 缺失接近死欄,保留註記 |
| housetype_905L / housingtype_772L | 類別 | 直取 | 居住型態;>96% 缺失接近死欄,保留註記 |
| safeguarantyflag_411L | 布林 | 直取 | 附加保障產品旗標,True/False 各半有變異 |
| persontype_1072L / persontype_792L / role_1084L / type_25L | 類別 | 直取 | 本人列的類型/角色代碼,弱訊號但保留 |
| contaddr_district_15M / registaddr_district_1083M / registaddr_zipcode_184M | 類別 | 直取 | 高基數地理遮罩欄,原值保留交由後續建模端決定編碼方式 |

### B. 關係人特徵(`filter(num_group1>0)` 後聚合)

| 特徵 | 計算 | 理由 |
|---|---|---|
| `n_related_persons` | `count(*)` | 申請時登錄的關係人數,反映可驗證性與家庭/社會連結結構 |
| `relationshiptoclient_415T_mode` / `_n_unique` | mode、n_unique(既有 `build_cat_agg` 模式) | 關係組成(配偶/父母/朋友…)與多樣性,以 415T 為主指標 |
| `relationshiptoclient_642T_mode` | mode | 補充版本(與 415T 部分重複,僅取眾數作交叉參考) |
| `persontype_792L_mode` | mode | 關係人類型組成 |
| `rel_addr_n_unique` | `registaddr_district_1083M` 於關係人列的 n_unique | 關係人戶籍區異質度(同戶 vs 分散各地),資訊可驗證性訊號 |

`remitter_829L` 恆為 `{False, null}`,不產出模型特徵;關係人列的收入/生日 100% 缺失,不聚合。

### C. person_2 指標二次聚合(7 來源欄 × 7 指標)

- **本人列直取**(`_appl` 後綴):對應源欄的 `mode`、`mode_ratio`、`n_unique`、`entropy` ——
  p2 列 92% 屬本人,聯絡方式/地址/雇主/經濟狀態的「本人紀錄組成」直取語意最直接
- **全案彙總**(跨所有 num_group1):`null_count` → `sum`、`non_null_count` → `sum`
  (資料完整度總量);`n_unique`、`entropy` → `max`(全案最異質的一人,資訊多樣性上界)
- 結構性缺失不於列層級補值,聚合統計天然跳過 null;僅 count/sum 類指標聚合後補 0

## 步驟 4:缺失處理原則(沿用 applprev 階段已驗證邏輯)

- 列層級不補值,聚合統計(mode/n_unique/entropy/mean 類)天然跳過 null。
- 聚合後僅 count/sum 類欄位補 0(語意=「無此紀錄」);本人直取欄與 mode 類欄保留 null,
  缺失本身即訊號,交由建模前依 `person_1_importance.csv` 既有填補建議統一處理。

## Verification

1. join 檔:列數 = 2,973,991;p2 欄非空列數 = 1,561,280;`date_decision` 無缺失。
2. agg 檔:列數 = 1,526,659(distinct case_id);`n_related_persons` 總和 = 1,447,332
   (= 關係人列總數,即 2,973,991 − 1,526,659)。
3. 抽驗:`age_years` 落於 (0,100) 合理區間、`tenure_years` 無負值(未來日已清除)、
   `mainoccupationinc_384A_appl` max = 190,000(截尾生效);抽 3 個 case_id 比對
   `_appl` 欄與原表 num_group1=0 列數值一致。
4. 旗標/計數欄(`n_related_persons`、`*_null_count_sum` 等)無 NULL。

### 實際執行結果(已完成)

| 檢查項 | 結果 |
|---|---|
| join 檔列數 | 2,973,991 ✓ |
| p2 欄非空列數 | 1,561,280 ✓;`date_decision` 無缺失 ✓ |
| agg 檔列數 | 1,526,659 ✓ |
| `n_related_persons` 總和 | 1,447,332 ✓ |
| `age_years_appl` 範圍 | 20.96 ~ 76.04(合理)✓ |
| `tenure_years_appl` 最小值 | 0.0(無負值,未來日已清除)✓ |
| `mainoccupationinc_384A_appl` 最大值 | 190,000.0(截尾生效)✓ |
| `childnum_185L_appl` 最大值 | **NULL**(全欄皆空,見上方發現) ⚠ |
| 旗標/計數欄(`n_related_persons`、`*_null_count_sum`、`*_n_unique_max`、`*_entropy_max`) | 皆無 NULL ✓ |

輸出:`data/train_person_2_join_1.parquet`(2,973,991×94)、
`data/train_person_agg.parquet`(1,526,659×91)。
