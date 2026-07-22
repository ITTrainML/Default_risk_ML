# Train/Test 建模特徵管線重建計畫（data/train/ + data/test/ → df_train / df_test）

## Context

先前的 `base_final.parquet` 只為 **train** 建置,且產生它的聚合程式(credit_bureau、static、
tax_registry)多數未進版控。現在要建立一條**統一、可重現**的 ETL 管線,對 `data/train/` 與
`data/test/` 的原始分區檔,用**相同程式**重建最終選定的建模特徵(約 82 欄 + 5 個衍生欄),
使 train 與 test 的特徵定義完全一致,供後續特徵工程/建模使用。

**已與使用者確認的決策**:
- **忽略 credit_bureau_b 系列(b1 與 b2)** → 捨棄 3 個 `b2_pmts_dpdvalue_108P_*` 特徵。
- credit_bureau_a_2 的自訂統計指標(trend/longest_good_streak/recent_time_key/weighted_avg/
  recomputed/duration 等)**依 `feature_name_mapping.csv` 的「計算方式」欄實作**(此檔已在磁碟)。
- **train 與 test 都從原始分區檔重建**(同一份程式),不重用舊的 base_final.parquet。
- **test 無 `target`**(Kaggle 測試集未標記):test 輸出 = case_id + WEEK_NUM + 特徵;train 保留 target。
- 輸出 **Polars DataFrame 變數 `df_train` / `df_test`**,並各自寫出 parquet。

**已從 git 歷史/磁碟回收的關鍵參考**:
- `feature_name_mapping.csv`(151 列,磁碟上)——credit_bureau_a_2 每個後贅詞的精確計算方式。
- `bureau_a2_merge_GroupRule.txt`、`bureau_a1_GroupRule.txt`、`bureau_a1_clean.txt`(git 歷史 commit 7721d01)——bureau 清洗/聚合規則。
- 既有 repo 程式:`data_preprocessing_2.py`(applprev_1 聚合+reject_rate/last_status/tenure_years)、
  `train_person_agg.py`(person_1 的 `_appl` 欄+age/tenure)——衍生欄公式來源(見步驟 5)。

## 檔案結構與 depth 分類（Home Credit 慣例 `{split}_{table}_{depth}_{partition}`）

`data/train/` 33 檔、`data/test/` 36 檔(test 的分區更多)。以檔名倒數第二個數字為 **depth(X)**,
最後一個數字為分區(Y);`file_name_W_X` 相同者(僅差分區 Y)先 **union**:

| depth | 表(union 後) | 處理方式 | 選中特徵數 |
|---|---|---|---|
| 0 | `static_0`(分區 0/1[/2])、`static_cb_0`、`base` | 清洗後直接 join(已是 case_id 層級) | static_0=27、static_cb_0=8 |
| 1 | `credit_bureau_a_1`(分區 0-3/0-4)、`applprev_1`(分區 0-1/0-2)、`tax_registry_a_1` | 一次聚合 by `case_id` | a_1=13、applprev_1=4、tax_a=2 |
| 1 | `person_1` | 取 num_group1==0(本人)直取 + 衍生 | person `_appl`=5 |
| 2 | `credit_bureau_a_2`(分區 0-10/0-11) | 一次聚合 by (case_id,num_group1) → 二次聚合 by case_id | a_2=20 |
| — | `credit_bureau_b_1/b_2`、debitcard、deposit、other、person_2、tax_b/c | **忽略**(無選中特徵或使用者指定忽略) | 0 |

## 步驟 1:union 分區檔

對每個 `file_name_W_X`,`pl.scan_parquet("data/{split}/{split}_{name}_{X}_*.parquet")` 惰性讀取後
`concat`(vertical)。單檔(無分區)直接讀。train/test 各自處理,前贅詞 `train_`/`test_`。

## 步驟 2:合理值清洗（依使用者 step 2,為本任務權威規則）

對「只讀取的必要欄位」套用(逐 split 各自以自身資料計算分位數,避免 test 用到 train 統計):
1. **連續型 99%PR 截尾(winsorize,不刪列)**:`x > p99 → p99`;**若 `p99 == 0` 則不截尾**
   (避免抹除稀疏逾期訊號,沿用 `data_preprocessing_2.py` 對 actualdpd 的既有決策)。
2. **大寫 `A` 結尾欄(金額)→ 負值設 NULL**(`col < 0 → null`)。
3. **大寫 `D` 結尾欄(日期)→ 不合理未來日設 NULL**:晚於該列 `date_decision`(person)或全域上界者設 NULL。
4. **bureau_a2 補充規則**(來自 `bureau_a2_merge_GroupRule.txt`):DPD(`P` 欄)負值設 NULL;
   繳款年月(`pmts_year_*T`)落在 2025-2028 等未來/無效年份設 NULL(月份須在 1-12)。

## 步驟 3:聚合（依欄位後贅詞決定統計指標,計算方式一律引用 `feature_name_mapping.csv`）

**後贅詞 → 統計指標對照(節錄自 `feature_name_mapping.csv` 計算方式欄)**:

| 後贅詞 | 計算方式 |
|---|---|
| `mean`/`max`/`min`/`median`/`std` | 非空值 mean/max/min/median、樣本標準差(stddev_samp) |
| `mode`/`n_unique`/`entropy` | 眾數 / count distinct / 分布熵 |
| `positive_count`/`overdue_rate`/`mean_positive`/`sum_positive` | x>0 筆數 / (x>0 筆數÷非空筆數) / x>0 之 mean / x>0 之 sum |
| `null_count`/`null_rate`/`non_null_count` | NULL 筆數 / NULL 比例 / 非 NULL 筆數 |
| `recent3_mean`/`recent6_mean`/`recent12_mean` | 依(年月鍵, num_group2)排序後最近 N 筆平均 |
| `trend` | 依時間排序後 `regr_slope`(線性回歸斜率,是否越來越嚴重) |
| `last`/`recent_time_key` | 排序後最新一筆值 / `year*12+month` 的最大有效值 |
| `consecutive_max`/`current_streak`/`longest_good_streak` | 連續 x>0 最長 / 最新往回連續 x>0 期數 / 連續 x==0 最長 |
| 日期 `min`/`max`/`n_unique`/`duration`/`recent` | 年月鍵 min/max、distinct 數、(max−min 跨度)、max |
| 二次(case 層級)`recomputed` | 以 Σsum_positive ÷ Σpositive_count 重算、並取 max |
| 二次 `weighted_avg` | 以非缺失筆數為權重的加權平均 |
| 二次 `mean_fallback`/`max_fallback` | 對一次聚合值取 mean/max,缺失以 fallback 值(0)替補 |

**depth-1(X=1)** — `aggregate by case_id`。credit_bureau_a_1/tax_registry_a_1 用上表對應後贅詞的
單層統計(選中欄後贅詞:`__min`/`__sum`/`__mean`/`__std`/`__max`/`__null_rate`)。

**depth-2(X=2,credit_bureau_a_2)** — 兩層,選中特徵名格式 `{raw}_{L1}__{L2}`:
1. **一次聚合** by `(case_id, num_group1)`:對原始欄算 L1 後贅詞統計。時間序列類(trend/recent_time_key/
   longest_good_streak/duration…)需先在群組內依 `年月鍵(year*12+month), num_group2` 排序。
   年月鍵取自各欄對應的繳款年月欄(active=`pmts_year_1139T`/`pmts_month_158T`;
   closed=`pmts_year_507T`/`pmts_month_706T`;`pmts_dpd_1073P`=active、`pmts_dpd_303P`=closed)。
2. **二次聚合** by `case_id`:對一次聚合結果算 L2 後贅詞統計(`__max`/`__mean`/`__sum`/`__min`/
   `__weighted_avg`/`__recomputed`/`__mean_fallback`/`__max_fallback`)。

實作:寫一個「特徵名 → (原始欄, L1 stat, L2 stat)」解析器 + 一組後贅詞統計函式庫(Polars 表達式),
只計算選中特徵所需的統計,不全量展開(避免產生上千無用欄)。categorical 欄(subjectroles_name_838M)
用 mode/n_unique/entropy。

## 步驟 4:join 回 base

以 union 後的 `{split}_base.parquet`(case_id、WEEK_NUM、date_decision、MONTH、train 另有 target)
為主表,`left join`(on `case_id`):depth-0 表(static_0、static_cb_0)、depth-1 聚合表、
depth-2 二次聚合表、步驟 5 衍生欄。最後 `select` 僅保留最終欄位清單。

## 步驟 5:5 個衍生欄（回憶自既有程式,公式記錄如下）

| 欄位 | 來源 | 公式(依 `case_id` 分組) |
|---|---|---|
| `reject_rate` | applprev_1 | `sum(status_219L=='D') / sum(status_219L is not null)`(見 [data_preprocessing_2.py](data_preprocessing_2.py)) |
| `last_status` | applprev_1 | 依 `creationdate_885D` 升冪排序後 `status_219L` 的 `last()` |
| `tenure_years_max` | applprev_1 | `tenure_years = 2024 − (year(employedfrom_700D)+(month−1)/12)`,取群組 `max` |
| `age_years_appl` | person_1 | 本人列(num_group1==0):`(date_decision − coalesce(birth_259D,birthdate_87D)).days / 365.25`(見 [train_person_agg.py](train_person_agg.py)) |
| `tenure_years_appl` | person_1 | 本人列:`(date_decision − empl_employedfrom_271D).days / 365.25` |

person_1 的 5 個 `_appl` 類別欄(education_927M_appl / incometype_1044T_appl / familystate_447L_appl /
empl_industry_691L_appl / registaddr_zipcode_184M_appl)= 本人列(num_group1==0)直取對應原始欄。

## 產出檔案

- **新增** `build_model_dataset.py`(repo 根目錄,Polars,風格沿用 `data_preprocessing_2.py`):
  以 `SPLIT` 參數(train/test)驅動同一套函式;`main()` 對 train、test 各跑一次。
  模組化函式:`union_partitions`、`clean_continuous/clean_by_label`、`agg_depth1`、
  `agg_depth2`(含後贅詞函式庫)、`derive_applprev`、`derive_person`、`join_to_base`。
- **新增** `data/df_train.parquet`、`data/df_test.parquet`;程式回傳 `df_train`、`df_test`(Polars)。
- **新增** `build_model_dataset_plan.md`(專案內,本計畫存查)。

## 驗證方式

1. 執行 `python build_model_dataset.py` 無錯誤;`df_train` 列數 = `train_base` 的 case_id 數
   (1,526,659)、`df_test` 列數 = `test_base` case_id 數;兩者欄位集合一致(test 少 `target`)。
2. 每個最終欄位皆存在;無因 join 產生的重複 case_id;`df_train`/`df_test` 欄名 = 選定清單。
3. 抽驗一致性:`df_train` 的 `reject_rate ∈ [0,1]`、`age_years_appl` 落在合理區間;
   挑 2-3 個 case_id 手工比對 depth-2 的 `pmts_dpd_303P_std__max`(先 (case_id,num_group1) 算 std、
   再 case_id 取 max)與原始列一致。
4. 清洗檢查:winsorize 欄 max ≈ 各自 split 的 p99;A 欄無負值;D 欄無未來日;p99==0 的欄未被截尾。
5. train vs test 欄位分布抽樣比對(數值範圍、缺失率)合理,無因分區/前贅詞錯置導致的整欄 NULL。
