# 建模資料集切分計畫(欄位保留 + Train/Validation Split)

## Context

前序已完成 `data/base_final.parquet` 的 WOE/IV 單變量篩選
([woe_iv_feature_selection_summary.md](woe_iv_feature_selection_summary.md))
與候選特徵的 VIF 多重共線性分析
([vif_multicollinearity_summary.md](vif_multicollinearity_summary.md))。
使用者現已根據這兩份分析人工選定最終要進入後續特徵工程/建模的欄位清單(87 個特徵 +
`case_id`/`target`/`WEEK_NUM`),需要:

1. 從 `data/base_final.parquet` 只保留這份欄位清單。
2. 依 `WEEK_NUM`(決策所屬週數)由大到小排序,取最新 20% 當驗證集(validation set),
   其餘當訓練集(train set),供後續模型開發使用(時間序列外推式驗證,避免用未來
   資料訓練、過去資料驗證的洩漏)。

**已與使用者確認的細節**:
- 使用者提供的清單中 `taget` 為 `target`(目標變數,1=違約、0=正常還款)的筆誤,
  已修正,其餘 87 個特徵欄位皆確認存在於 `base_final.parquet` schema 中。
- **20% 驗證集的計算基準**:以 **相異 WEEK_NUM 值的數量** 為準(而非列數)。
  `WEEK_NUM` 值域 0~91,共 92 個相異週;`round(92 × 0.2) = 18`,故取數值最大的
  **18 週**(`WEEK_NUM ∈ [74, 91]`)當驗證集,其餘 `WEEK_NUM ∈ [0, 73]`(74 週)當訓練集。
  已用 DuckDB 驗證此切法下驗證集實際列數為 **194,658 列(佔全部 1,526,659 列的
  12.75%)**——因各週列數並非均勻分布(每週介於 825~17,886 列不等),此切法下驗證集
  的「列數佔比」明顯低於 20%,但使用者已確認是以「相異週數」而非「列數」為準,
  此為預期結果。
- **輸出格式**:兩個獨立 parquet 檔 ——`data/train_model.parquet`(訓練集)、
  `data/valid_model.parquet`(驗證集),欄位結構相同(即保留清單的 90 欄)。

## 最終保留欄位清單(90 欄)

- 鍵值/目標/時間:`case_id`、`target`(修正自 `taget`)、`WEEK_NUM`
- 其餘 87 個特徵欄位(使用者提供,已逐一比對 `base_final.parquet` schema 確認存在),
  涵蓋:逾期/DPD 類(`avgdpdtolclosure24_3658938P`、`maxdbddpdtollast12m_3658940P`、
  `pctinstlsallpaidlate1d_3546856L`、`dpdmax_757P__null_rate` 等)、申請歷史類
  (`reject_rate`、`numrejects9m_859L`、`last_status`、`lastrejectreason_759M` 等)、
  人口/所得類(`age_years_appl`、`tenure_years_appl`、`tenure_years_max`、
  `incometype_1044T_appl`、`education_927M_appl` 等)、信用機構 b1/b2 逾期率類
  (`b2_pmts_dpdvalue_108P_overdue_rate_recomputed` 等)、金額/合約類
  (`disbursedcredamount_1113A`、`totalamount_6A__sum`、`maxdebt4_972A` 等)。

## 產出檔案

- **新增** `build_train_valid_dataset.py`(repo 根目錄,Polars 風格沿用
  [data_preprocessing_2.py](data_preprocessing_2.py))。
- **新增** `data/train_model.parquet`、`data/valid_model.parquet`。

## 方法設計

### 步驟 1:欄位清單與型別確認

腳本內以常數 `KEEP_COLS`(90 欄,見上)硬編碼欄位清單(來源:使用者本次提供,
已排除筆誤 `taget`)。執行前以 DuckDB `DESCRIBE` 交叉比對 `base_final.parquet` schema,
若有任何欄位不存在則直接報錯中止(fail-fast),避免靜默漏欄。

### 步驟 2:讀取並篩選欄位

以 `pl.scan_parquet('data/base_final.parquet').select(KEEP_COLS)` 惰性讀取只需要的
90 欄(避免載入其餘 ~1,110 欄,降低記憶體壓力)。

### 步驟 3:依 WEEK_NUM 切分 train/valid

1. 計算相異 `WEEK_NUM` 值集合,取數值最大的 18 個(`round(n_distinct_weeks * 0.2)`,
   目前資料為 92 週 → 18 週,即 `WEEK_NUM >= 74`)。
2. `valid = df.filter(pl.col("WEEK_NUM") >= cutoff)`、
   `train = df.filter(pl.col("WEEK_NUM") < cutoff)`。
   `cutoff` 由程式動態計算(依「相異週數 × 20% 四捨五入」),不寫死 74,以便未來
   資料更新(週數範圍改變)時仍正確。

### 步驟 4:寫出與驗證性輸出

- `train.write_parquet('data/train_model.parquet')`
- `valid.write_parquet('data/valid_model.parquet')`
- stdout 印出:兩檔案的 shape、WEEK_NUM 範圍、`target` 違約率(train vs valid 分別
  印出,供人工檢查切分後違約率是否仍相近、有無明顯 population drift)。

## 驗證方式

1. 執行 `python build_train_valid_dataset.py`,無錯誤。
2. `train.height + valid.height == 1,526,659`(與 `base_final.parquet` 總列數一致)。
3. `train` 的 `WEEK_NUM` 最大值 < `valid` 的 `WEEK_NUM` 最小值(確認切分無交錯)。
4. 兩檔案欄位數皆為 90、欄名與 `KEEP_COLS` 完全一致。
5. `valid` 列數 ≈ 194,658(±因未來資料更新而變動,但邏輯應可重現此值於當前資料)。
6. 印出並人工檢查 train/valid 的 `target` 違約率,確認無異常斷崖(週數切分下違約率
   有自然漂移屬預期,不代表錯誤)。
