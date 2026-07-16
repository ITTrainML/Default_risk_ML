# applprev_1_join 清洗與聚合計畫

## Context

`data/train_applprev_join.parquet`(6,525,979 列、1,221,522 個 case_id)是先前申請紀錄 applprev_1 與 applprev_2 聚合結果的 join(由 `data_preprocessing_1.py` 產出)。本次要對其中 15 個指定欄位做清洗、缺失值處理,並以 `case_id` 為 key 聚合成 case 層級特徵表,供後續違約風險模型使用。

**產出**:新腳本 `data_preprocessing_2.py`(Polars,風格沿用 `data_preprocessing.py`),輸出 `data/train_applprev_1_agg.parquet`。

## 探索結論(已用 DuckDB 驗證)

| 欄位 | 型別 | 缺失率 | 類型判定 | 說明(feature_definitions.csv) |
|---|---|---|---|---|
| actualdpd_943P | DOUBLE | 0.04% | 連續型 | 先前合約實際逾期天數;**p99=0、max=4206** |
| byoccupationinc_3656910L | DOUBLE | 76.5% | 連續型 | 先前申請的職業收入 |
| cancelreason_3545846M | VARCHAR | 0%* | 離散型(76 類) | 申請取消原因 |
| credacc_credlmt_575A | DOUBLE | 3.0% | 連續型 | 信用卡額度 |
| currdebt_94A | DOUBLE | 34.4% | 連續型 | 先前申請目前債務 |
| education_1138M | VARCHAR | 0%* | 離散型(6 類) | 教育程度 |
| employedfrom_700D | VARCHAR(日期字串) | 59.6% | 日期型 | 就業起始日;56 列晚於 creationdate_885D |
| mainoccupationinc_437A | DOUBLE | 1.6% | 連續型 | 主要收入金額 |
| maxdpdtolerance_577P | DOUBLE | 47.4% | 連續型 | 最大容忍逾期天數;p99=329 |
| outstandingdebt_522A | DOUBLE | 34.6% | 連續型 | 未償債務 |
| profession_152M | VARCHAR | 0%* | 離散型(11,508 類,98.9% 為佔位值) | 職業 |
| rejectreason_755M | VARCHAR | 0%* | 離散型(18 類) | 拒絕原因 |
| rejectreasonclient_4145042M | VARCHAR | 0%* | 離散型(14 類) | 客戶端拒絕原因 |
| revolvingaccount_394A | DOUBLE | 95.5% | **實為帳戶編號**(值域 5.4 億~8.0 億) | 循環帳戶 |
| status_219L | VARCHAR | 0.001% | 離散型(11 類) | 先前申請狀態 |

\* M 欄位無 NULL,但 `'a55475b1'` 為資料集的雜湊「缺失/無」佔位類別(cancelreason 68%、rejectreason 71%、rejectreasonclient 76%、education 47%、profession 99%)。

**status_219L 語意驗證**(交叉比對 approvaldate / rejectreason / cancelreason):
- `D`(40.9%)= **被拒絕**:0% 有 approvaldate、61% 有 rejectreason
- `K`(40.7%)= 已核准且已結案:100% approvaldate、67% 有最後還款日
- `A`(11.0%)= 已核准使用中:100% approvaldate + activation、0% 最後還款
- `T`(6.8%)= 取消:92% 有 cancelreason
- 其餘 N/Q/S/L/H/P/R 合計 <0.7%

**排序依據**:`num_group1` 遞增 ≈ creationdate 遞減(num_group1=0 為最近一次申請),但只有 89% 一致 → 需要「最近一次」的統計時,**以 `creationdate_885D` 升冪排序取 last**(缺失率僅 0.001%),不用 num_group1。

**缺失結構驗證**:currdebt / outstandingdebt / maxdpdtolerance 的缺失集中於 `D`(70~99.8%)與 `T`(82~92%)狀態,`K`/`A` 幾乎為 0% → 屬「合約不存在」的結構性缺失。

## 步驟 1:欄位篩選

從 join 檔只讀 `case_id` + `creationdate_885D`(排序用)+ 15 個指定欄位。

## 步驟 2:清洗規則實作

1. **99%PR 截尾(winsorize,不刪列)** — 對連續型欄位,超過全體(非空)99% 分位數者以 p99 取代:
   - 套用:`byoccupationinc_3656910L`、`credacc_credlmt_575A`、`currdebt_94A`、`mainoccupationinc_437A`、`maxdpdtolerance_577P`、`outstandingdebt_522A`
   - **不套用 `actualdpd_943P`**(使用者決議:p99=0 會抹掉全部逾期訊號,此欄不做離群值處理)
   - **不套用 `revolvingaccount_394A`**(帳戶編號,改為二元旗標,見步驟 3)
2. **大寫 A 結尾 → 刪除負值** — 實測所有 A/P 數值欄 min=0,無負值;仍實作 `filter(col >= 0 | col.is_null())` 作為防護
3. **大寫 D 結尾 → 未來日處理** — `employedfrom_700D` 以**該筆申請的 `creationdate_885D` 為基準**:就業起始日晚於申請日不合理(56 列),**設為 NULL**(保留該列其他欄位)
4. **就業年資衍生欄**(使用者需求):`tenure_years = 2024 - (year(employedfrom_700D) + (month(employedfrom_700D)-1)/12)`,即以 2024 年為基準減去該欄年月

## 步驟 3:缺失值處理(逐欄理由)

| 欄位 | 補值方式 | 理由 |
|---|---|---|
| actualdpd_943P | 補 0 | 缺失率僅 0.04%;DPD 缺失 = 無逾期紀錄,語意上等同 0 |
| maxdpdtolerance_577P | 補 0 | 缺失集中於 D/T(被拒/取消,無合約即無逾期),結構性缺失補 0 |
| currdebt_94A | 補 0 | 同上:D 狀態 70.6%、T 狀態 81.8% 缺失,K/A 為 0% — 無合約即無債務 |
| outstandingdebt_522A | 補 0 | 同上(D 70.6%、T 84.4% 缺失) |
| credacc_credlmt_575A | 補 0 | 無信用卡帳戶即無額度,補 0 符合語意 |
| mainoccupationinc_437A | **不於列層級補值** | 收入不可能為 0,缺失率僅 1.6%;聚合統計(mean/max)天然忽略 null,補中位數反而在 case 內注入假值 |
| byoccupationinc_3656910L | **不補值** + 聚合時輸出缺失率特徵 | 缺失 76.5%,任何單值補法都會扭曲分布;缺失本身可能有訊號(如未申報),以 `null_ratio` 保留 |
| employedfrom_700D / tenure_years | **不補值** | 日期無合理填補值;聚合統計忽略 null,另輸出缺失率 |
| revolvingaccount_394A | 轉二元 `has_revolving = is_not_null()`,null→0 | 使用者決議:值為帳戶編號,數值統計無意義;「是否曾有循環帳戶」才是訊號 |
| 5 個 M 類別欄 | 不補(無 NULL);`'a55475b1'` 保留為獨立類別 | 佔位值本身即「無此原因/未填」的類別,對眾數/熵等統計有意義 |
| status_219L | 66 筆 NULL 不補,計算比率時從分母排除 | 佔比 0.001%,任何補法都是雜訊 |

## 步驟 4:以 case_id 聚合(逐欄分析與指標理由)

**排序**:聚合前先 `sort(["case_id", "creationdate_885D"])`,使 `last()` = 最近一次申請。

### 連續型

| 欄位 | 指標 | 理由 |
|---|---|---|
| actualdpd_943P | `max`、`mean`、`sum(>0)` 次數(逾期次數) | max 抓最嚴重逾期(風險上界);mean 抓慣性;逾期「次數」比金額分布更穩健(此欄 99% 為 0,計數特徵最有效) |
| maxdpdtolerance_577P | `max`、`mean` | 同為 DPD 類,max 為歷史最差表現 |
| currdebt_94A | `sum`、`max`、`last` | sum = 跨先前申請的總現有債務(總負債水位);last = 最近一筆的債務現況 |
| outstandingdebt_522A | `sum`、`max`、`last` | 同上,未償債務總額直接對應償債壓力 |
| credacc_credlmt_575A | `max`、`last` | 額度反映授信評價,取歷史最高與最近值;sum 無意義(額度非流量) |
| mainoccupationinc_437A | `last`、`max`、`mean` | **last 最重要**(最近申報收入最接近現況,需排序);max/mean 抓收入水準與波動 |
| byoccupationinc_3656910L | `last`、`mean`、`null_ratio` | 同上;缺失率 76.5%,null_ratio 本身作為特徵 |

### 日期型

| 欄位 | 指標 | 理由 |
|---|---|---|
| employedfrom_700D → tenure_years | `sum`(使用者指定)、**建議另加 `last` 與 `max`** | **sum 的問題**:總年資 = Σ(每筆申請的年資),與「申請次數」高度共線 — 申請 5 次、每次年資 2 年的 sum=10,會比申請 1 次年資 8 年的人「看起來」更資深,混淆了穩定度與申請頻率兩種訊號。仍依需求輸出 sum,但**建議以 `last`(最近一次申請時的年資,最能代表現況)與 `max`(歷史最長年資,就業穩定度)為主力特徵**;模型可自行取捨 |

### 離散型(5 個 M 欄)

沿用 `data_preprocessing.py` 已驗證的模式:每欄輸出 `mode`、`mode_ratio`、`n_unique`、`entropy`、`null_count`(此處 M 欄無 null,可省略 null_count 或以 `'a55475b1'` 計數取代):

| 欄位 | 額外指標 | 理由 |
|---|---|---|
| education_1138M | `last` | 教育程度會隨時間更新,最近值最準 |
| profession_152M | 僅 `n_unique` + `'a55475b1'` 比例 | 11,508 類且 98.9% 為佔位值,mode 幾乎恆為佔位值,無鑑別力;職業「變動次數」與「有無申報」較有訊號 |
| cancelreason_3545846M | `mode`、`n_unique`、非佔位值比例 | 取消原因分布反映申請行為模式 |
| rejectreason_755M / rejectreasonclient_4145042M | 同上 | 拒絕原因與 status=D 高度相關,佐證拒絕率特徵 |

### status_219L 與拒絕率特徵(使用者指定需求)

已驗證 `D` = 被拒絕(0% approvaldate、61% rejectreason)。**計算方式**:

```
reject_rate = count(status_219L == 'D') / count(status_219L is not null)   # 依 case_id 分組
```

即:該客戶所有先前申請中,被拒絕件數 ÷ 有效狀態件數(66 筆 NULL 排除於分母)。另輸出配套特徵:

- `n_prev_apps = count(*)`:先前申請總數(拒絕率的分母規模,1 次被拒 100% 與 10 次被拒 100% 意義不同)
- `n_rejected = count(status == 'D')`:被拒次數
- `last_status`:最近一次申請狀態(需先按 creationdate 排序;最近被拒是強風險訊號)
- `cancel_rate = count(status == 'T') / count(not null)`:取消率(行為特徵,與拒絕率互補)

## 步驟 5:新腳本結構(data_preprocessing_2.py)

```
SRC = "data/train_applprev_join.parquet"
OUT = "data/train_applprev_1_agg.parquet"

1. pl.read_parquet 只讀 17 欄(case_id + creationdate_885D + 15 指定欄)
2. 型別轉換:employedfrom_700D / creationdate_885D → pl.Date
3. 清洗:A 欄負值過濾 → employedfrom 未來日設 NULL → tenure_years 衍生
   → 6 個連續欄 99%PR 截尾 → revolvingaccount 轉 has_revolving
4. 缺失補值(依步驟 3 表格)
5. sort(["case_id", "creationdate_885D"]) 後 group_by("case_id").agg(...)
6. write_parquet(OUT),印出 shape 與 columns
```

## 驗證方式

1. 執行 `python data_preprocessing_2.py`,確認無錯誤、輸出列數 = 1,221,522(distinct case_id)
2. 用 DuckDB 抽驗:
   - `reject_rate` 介於 [0,1];挑 2~3 個 case_id 手工比對 `D` 件數/總件數
   - 截尾欄位 max ≈ 全體 p99;actualdpd_943P 的 max 仍為 4206(未截尾)
   - tenure_years 無負值(未來日已清除);has_revolving 僅 {0,1}
3. 確認補 0 欄位在聚合後無 NULL;不補值欄位(收入類)聚合後 NULL 數合理

---

# 增量計畫:applprev_2_agg 欄位納入 case 層級聚合

## Context

第一階段(上述)已完成並驗證:`data_preprocessing_2.py` 輸出 `data/train_applprev_1_agg.parquet`(1,221,522 列 × 57 欄)。

**本次需求**:`train_applprev_join.parquet` 中還有 15 個來自 `train_applprev_2_agg.parquet` 的欄位——3 個原始類別欄 × 5 個指標(`mode / mode_ratio / n_unique / entropy / null_count`),由 `data_preprocessing.py` 在 `(case_id, num_group1)` 層級產出,即「**每筆先前申請**」一列。要將這些已聚合過的指標欄**二次聚合**到 `case_id` 層級,分析各欄類型、建議統計指標並說明理由。

**產出**:修改 `data_preprocessing_2.py` 重新輸出 `data/train_applprev_1_agg.parquet`。

## 探索結論(已用 DuckDB 驗證,含 case 層級變異檢查)

### 3 個來源欄位的語意與訊號強度

| 來源欄位 | 說明(feature_definitions.csv) | 關鍵發現 |
|---|---|---|
| cacccardblochreas_147M | 卡片凍結原因 | `_mode` 99.9% 為佔位值 `a55475b1`;`_entropy` 全表僅 6,397 列 > 0(0.1%);`_n_unique` 99.9% = 1 → **統計指標全是死特徵,只有「曾出現真實凍結原因」有訊號**(4,674 個 case,0.38%,稀有但為強負面信用事件) |
| conts_type_509L | 先前申請的聯絡方式類型 | `_mode` 分布豐富(PRIMARY_MOBILE 42%、EMPLOYMENT_PHONE 26%、HOME_PHONE 19%...);`_n_unique` 1~6;case 層級聚合變異充足(n_unique_max std=0.77、mode_ratio_mean std=0.20、entropy_max std=0.36)→ **訊號最豐富,套用完整指標組** |
| credacc_cards_status_52L | 先前信用帳戶的卡片狀態 | 整組 95.15% NULL(該筆申請無卡);251,163 個 case(20.6%)曾持卡;有值時 mode = CANCELLED(49%)/ACTIVE(35%)/INACTIVE(16%)/BLOCKED(0.6%);曾 CANCELLED/BLOCKED 的 case 有 131,703 個(10.8%);非空樣本中 `_n_unique` 94.6% = 1 → **「有無卡」與「負面狀態」是主訊號,ratio/entropy 無變異** |

### 5 類指標欄的類型判定

| 指標欄 | 類型 | 值域/分布 |
|---|---|---|
| `_mode` | 離散型(類別) | 每筆申請內的眾數類別 |
| `_mode_ratio` | 連續型 | (0,1],天然有界 → 無須 99%PR 截尾 |
| `_n_unique` | 離散型(計數) | 1~6,天然有界 |
| `_entropy` | 連續型 | ≥0,normalize 後有界 |
| `_null_count` | 離散型(計數) | case 層級 sum 後 p99=17、max=46,值域合理 → 無須截尾 |

**缺失結構**:`_mode/_mode_ratio/_n_unique/_entropy` 的 NULL(cacccard 0.89%、conts 0.76%、credacc 95.15%)= 該筆申請在 applprev_2 整組無紀錄,屬結構性缺失 → **不補值**(聚合統計天然忽略,`has_*` 旗標顯式編碼「有無紀錄」);`_null_count` 僅各 1 列 NULL → 補 0。

## 二次聚合設計(group by case_id,聚合前已按 creationdate_885D 排序)

### conts_type_509L(完整指標組)

| 新特徵 | 計算 | 理由 |
|---|---|---|
| `conts_type_509L_mode_last` | 排序後 `last()` | 最近一次申請的主要聯絡方式,最貼近申請當下狀態(**需先排序**) |
| `conts_type_509L_mode_mode` / `_mode_ratio` / `_n_unique` | 對 `_mode` 欄套用既有 `build_cat_agg`(value-counts 模式) | mode of modes = 跨申請的長期慣用聯絡方式;其 mode_ratio = **跨申請的偏好穩定度**(每次都留同一種 vs 常換);n_unique = 歷史上換過幾種主要聯絡方式 |
| `conts_type_509L_mode_ratio_mean` / `_min` | `mean()`、`min()` | mean = 申請內聯絡方式集中度的平均;min = 最分散的一次(異質性下界)。不取 sum(比率加總無意義) |
| `conts_type_509L_n_unique_max` / `_mean` | `max()`、`mean()` | max = 單次申請留過最多幾種聯絡方式;mean = 平均多樣性。多樣性反映資料完整度與可驗證性 |
| `conts_type_509L_entropy_max` / `_mean` | `max()`、`mean()` | 與 n_unique 互補:n_unique 只算「幾種」,entropy 衡量「分布多均勻」;max 抓最異質的一次申請 |
| `conts_type_509L_null_count_sum` | `sum()` | 跨所有先前申請的聯絡方式缺失總筆數 = 客戶資料完整度;資訊不透明本身可能是風險訊號 |

### cacccardblochreas_147M(特化為事件旗標,避免死特徵)

| 新特徵 | 計算 | 理由 |
|---|---|---|
| `has_card_block_any` | `any(mode != 'a55475b1')` → 0/1 | 曾有真實卡片凍結原因 = 強負面信用事件;稀有(0.38%)但事件型旗標對樹模型仍有效 |
| `n_card_block_apps` | `count(mode != 'a55475b1')` | 凍結事件出現在幾筆申請(重複發生比單次更嚴重) |
| `cacccardblochreas_147M_null_count_sum` | `sum()` | 資料完整度 |

不對此欄做 mode_ratio/entropy/n_unique 統計:99.9% 列無變異,產出的特徵無鑑別力。

### credacc_cards_status_52L(先建持卡旗標,狀態指標只在有卡時有意義)

| 新特徵 | 計算 | 理由 |
|---|---|---|
| `has_card_any` | `any(mode is not null)` → 0/1 | 曾持卡(20.6% case);95.15% 結構性缺失下,「有無卡」本身就是特徵 |
| `n_card_apps` | `count(mode is not null)` | 有卡紀錄的申請筆數(持卡歷史深度) |
| `card_status_last` | `drop_nulls().last()`(排序後) | 最近一筆有卡申請的狀態;最近狀態最能代表現況(**需先排序**) |
| `card_cancelled_or_blocked_any` | `any(mode in ('CANCELLED','BLOCKED'))` → 0/1 | 曾有卡被取消/凍結(10.8% case)= 明確負面訊號,且 CANCELLED 佔有值樣本 49%,盛行率健康 |
| `credacc_cards_status_52L_null_count_sum` | `sum()` | 資料完整度 |

不做 mode_ratio/entropy 統計:有值樣本中 n_unique 94.6% = 1,無變異。

**設計原則說明**:二次聚合的 mean 是「每筆申請等權」而非「每筆聯絡紀錄等權」——這是刻意的:case 層級特徵應以「申請」為分析單位,申請內紀錄多寡不應影響權重。

## 實作:修改 data_preprocessing_2.py

1. `KEEP_COLS` 加入 15 個 a2 欄位
2. 3 個 `_null_count` 欄 `fill_null(0)`(各僅 1 列 NULL)
3. 既有 `group_by("case_id").agg(...)` 追加上表的聚合表達式(排序已就緒,`last()` 直接可用;`card_status_last` 用 `pl.col(...).drop_nulls().last()`)
4. `conts_type_509L_mode` 的跨申請眾數沿用既有 `build_cat_agg`(次數相同依類別值排序,結果可重現);此處來源是 L 欄無佔位值,`build_cat_agg` 增加參數略過 `non_placeholder_ratio`(否則恆為 1 的死特徵)
5. 重新 `write_parquet(OUT)`

## 驗證方式(增量)

1. 執行 `python data_preprocessing_2.py`:無錯誤、輸出列數 = 1,221,522、原有 57 欄仍在
2. DuckDB 抽驗新增欄位:
   - 旗標欄僅 {0,1};`has_card_block_any` 的 case 數 ≈ 4,674、`has_card_any` ≈ 251,163、`card_cancelled_or_blocked_any` ≈ 131,703(與探索數據對齊)
   - `conts_type_509L_mode_ratio_mean` ∈ (0,1];`n_unique_max ≥ n_unique_mean`
   - 挑 2~3 個 case_id 與原 join 檔手工比對 `null_count_sum` 與 `card_status_last`
   - 既有欄位(reject_rate 等)與第一階段輸出抽樣比對,數值不變
