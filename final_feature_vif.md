# data/final_feature.csv 候選特徵 VIF 多重共線性報告
## Context
團隊以多種方法(AUC+KS、Catboost、EBM、GBM、TOP150 入選-1~5、WOE)彙整出 `data/final_feature.csv` 195 個候選特徵。本報告排除非特徵中繼欄 `date_decision` 後,對其餘 194 個候選特徵計算 VIF (Variance Inflation Factor),檢查多重共線性,供後續人工決定去留(本報告為 report-only,不自動剔除欄位)。
## 方法
1. 以 DuckDB reservoir sampling(固定 seed=42)從 `data/base_final.parquet` 取出 500,000 列,所有候選欄共用同一批列以保留欄位間相關結構。
2. 每欄以與 `feature_selection_woe_iv.py` 相同的決策樹分箱(數值/時間戳)或類別稀有併箱(類別欄)+ `category_encoders.WOEEncoder` 轉換為 WOE 數值欄,組成 500,000 × 194 的 WOE 矩陣。
3. VIF 以相關矩陣虛擬逆(`numpy.linalg.pinv`)的對角線計算(與加入常數項的逐欄 OLS 迴歸法數學等價,但快得多);另抽樣欄位以 `statsmodels.variance_inflation_factor` 獨立重算比對一致性(見下方驗證章節)。
## 分層門檻
| VIF 區間 | 層級 |
|---|---|
| < 5 | 低(可接受) |
| 5 – 10 | 中(留意) |
| > 10 | 高(建議剔除) |

## 結果摘要
- 分析欄位總數:194

| 層級 | 欄位數 |
|---|---|
| 高(建議剔除) | 104 |
| 低(可接受) | 58 |
| 中(留意) | 32 |

## 驗證:statsmodels 交叉比對
| column | corr-inverse VIF | statsmodels VIF |
|---|---|---|
| pmtamount_36A_positive_count | 155029.8311 | 155029.8310 |
| pmts_dpd_303P_mean__max | 39.8531 | 39.8531 |
| pmts_overdue_1152A_std__mean | 11.2548 | 11.2548 |
| maxdpdlast3m_392P | 4.3469 | 4.3469 |
| maxannuity_4075009A | 1.0082 | 1.0082 |

## 完整 VIF 排名
| column | source | var_type | VIF | tier |
|---|---|---|---|---|
| pmtamount_36A_positive_count | TOP150入選-2 | numeric | 155029.8311 | 高(建議剔除) |
| pmtamount_36A_non_null_count | Catboost | numeric | 38756.7492 | 高(建議剔除) |
| tax_registry_c_count | TOP150入選-2 | numeric | 38756.7492 | 高(建議剔除) |
| amount_4527230A_positive_count | Catboost | numeric | 27493.0546 | 高(建議剔除) |
| tax_registry_a_count | TOP150入選-2 | numeric | 27487.7593 | 高(建議剔除) |
| relationshiptoclient_415T_mode | TOP150入選-2 | categorical | 2036.9975 | 高(建議剔除) |
| relationshiptoclient_642T_mode | TOP150入選-2 | categorical | 2036.9198 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed | AUC+KS | numeric | 888.1976 | 高(建議剔除) |
| b2_pmts_dpdvalue_108P_overdue_rate_recomputed | AUC+KS | numeric | 887.5829 | 高(建議剔除) |
| days360_512L | TOP150入選-2 | numeric | 158.6245 | 高(建議剔除) |
| numberofqueries_373L | Catboost | numeric | 156.8122 | 高(建議剔除) |
| dpdmax_139P__max | TOP150入選-1 | numeric | 142.5017 | 高(建議剔除) |
| prolongationcount_599L__sum | TOP150入選-1 | numeric | 138.1013 | 高(建議剔除) |
| prolongationcount_599L__mean | GBM | numeric | 127.8055 | 高(建議剔除) |
| pmts_dpd_1073P_positive_count__max | TOP150入選-1 | numeric | 119.5198 | 高(建議剔除) |
| pmts_dpd_1073P_sum_positive__max | TOP150入選-1 | numeric | 115.0709 | 高(建議剔除) |
| dpdmax_139P__sum | TOP150入選-3 | numeric | 112.8040 | 高(建議剔除) |
| pmts_dpd_1073P_positive_count__sum | TOP150入選-1 | numeric | 104.9287 | 高(建議剔除) |
| pmts_dpd_1073P_sum_positive__sum | TOP150入選-1 | numeric | 102.3072 | 高(建議剔除) |
| overdueamountmax_155A__sum | TOP150入選-2 | numeric | 101.6908 | 高(建議剔除) |
| pmts_overdue_1140A_sum_positive__max | TOP150入選-2 | numeric | 100.7600 | 高(建議剔除) |
| overdueamountmax2_14A__sum | TOP150入選-1 | numeric | 97.6194 | 高(建議剔除) |
| pmts_overdue_1140A_sum_positive__sum | TOP150入選-3 | numeric | 91.9105 | 高(建議剔除) |
| overdueamountmax_155A__mean | TOP150入選-4 | numeric | 75.1624 | 高(建議剔除) |
| numberofoverdueinstlmax_1039L__sum | TOP150入選-3 | numeric | 72.6657 | 高(建議剔除) |
| overdueamountmax2_14A__max | TOP150入選-2 | numeric | 72.3234 | 高(建議剔除) |
| numberofoverdueinstlmax_1039L__max | TOP150入選-3 | numeric | 68.9810 | 高(建議剔除) |
| overdueamountmax_155A__max | TOP150入選-2 | numeric | 68.1557 | 高(建議剔除) |
| numberofoverdueinstlmax_1151L__max | TOP150入選-2 | numeric | 64.8706 | 高(建議剔除) |
| dpdmax_757P__max | TOP150入選-2 | numeric | 63.8162 | 高(建議剔除) |
| numberofoverdueinstlmax_1151L__sum | TOP150入選-2 | numeric | 61.5314 | 高(建議剔除) |
| pmts_overdue_1140A_mean__max | WOE | numeric | 61.5306 | 高(建議剔除) |
| pmts_dpd_1073P_mean__mean | TOP150入選-1 | numeric | 59.7128 | 高(建議剔除) |
| pmts_overdue_1140A_std__max | TOP150入選-2 | numeric | 59.4908 | 高(建議剔除) |
| dpdmax_139P__mean | TOP150入選-4 | numeric | 58.7303 | 高(建議剔除) |
| dpdmax_757P__sum | TOP150入選-2 | numeric | 56.8406 | 高(建議剔除) |
| pmts_dpd_1073P_mean__max | TOP150入選-1 | numeric | 56.3269 | 高(建議剔除) |
| pmts_dpd_1073P_max__max | TOP150入選-1 | numeric | 54.6489 | 高(建議剔除) |
| pmts_overdue_1140A_std__mean | TOP150入選-3 | numeric | 54.1177 | 高(建議剔除) |
| pmts_overdue_1140A_mean__mean | TOP150入選-4 | numeric | 52.1347 | 高(建議剔除) |
| b2_pmts_dpdvalue_108P_overdue_rate_contract_max | AUC+KS | numeric | 49.3453 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_overdue_rate_contract_max | AUC+KS | numeric | 48.2880 | 高(建議剔除) |
| pmts_dpd_303P_recent12_mean__max | WOE | numeric | 47.4555 | 高(建議剔除) |
| pmts_dpd_1073P_std__mean | TOP150入選-4 | numeric | 47.1172 | 高(建議剔除) |
| pmts_dpd_1073P_recent12_mean__max | WOE | numeric | 45.0530 | 高(建議剔除) |
| overdueamountmax2_14A__mean | TOP150入選-2 | numeric | 44.9873 | 高(建議剔除) |
| pmts_dpd_1073P_recent12_mean__mean | WOE | numeric | 42.3986 | 高(建議剔除) |
| numberofoverdueinstlmax_1039L__mean | TOP150入選-4 | numeric | 40.1148 | 高(建議剔除) |
| pmts_dpd_303P_mean__max | TOP150入選-1 | numeric | 39.8531 | 高(建議剔除) |
| pmts_dpd_1073P_n_unique__max | TOP150入選-1 | numeric | 38.3925 | 高(建議剔除) |
| overdueamountmax_35A__mean | TOP150入選-2 | numeric | 38.0957 | 高(建議剔除) |
| pmts_dpd_303P_max__max | TOP150入選-1 | numeric | 37.3312 | 高(建議剔除) |
| pmts_dpd_303P_mean__mean | TOP150入選-3 | numeric | 36.3870 | 高(建議剔除) |
| pmts_dpd_303P_recent12_mean__mean | TOP150入選-2 | numeric | 35.7230 | 高(建議剔除) |
| pmts_dpd_303P_recent6_mean__max | TOP150入選-2 | numeric | 34.2103 | 高(建議剔除) |
| pmts_dpd_1073P_overdue_rate__weighted_avg | TOP150入選-3 | numeric | 33.5941 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_mean_contract_max | AUC+KS | numeric | 33.1214 | 高(建議剔除) |
| residualamount_856A__max | TOP150入選-2 | numeric | 32.6611 | 高(建議剔除) |
| numberofoverdueinstlmax_1151L__std | TOP150入選-3 | numeric | 32.3967 | 高(建議剔除) |
| pmts_dpd_303P_recent6_mean__mean | TOP150入選-2 | numeric | 32.2720 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_mean_weighted | AUC+KS | numeric | 31.9293 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_mean_contract_mean | AUC+KS | numeric | 31.2208 | 高(建議剔除) |
| pmts_dpd_1073P_std__max | TOP150入選-2 | numeric | 30.2417 | 高(建議剔除) |
| dpdmax_757P__std | TOP150入選-4 | numeric | 30.1649 | 高(建議剔除) |
| residualamount_856A__sum | TOP150入選-2 | numeric | 29.4215 | 高(建議剔除) |
| overdueamountmax2_398A__mean | TOP150入選-1 | numeric | 28.9306 | 高(建議剔除) |
| overdueamountmax2_398A__std | TOP150入選-1 | numeric | 28.2170 | 高(建議剔除) |
| numberofoverdueinstlmax_1151L__mean | TOP150入選-4 | numeric | 27.9832 | 高(建議剔除) |
| dpdmax_757P__mean | TOP150入選-4 | numeric | 27.7051 | 高(建議剔除) |
| pmts_dpd_1073P_consecutive_max__max | TOP150入選-2 | numeric | 27.4478 | 高(建議剔除) |
| overdueamountmax_35A__std | TOP150入選-2 | numeric | 25.1011 | 高(建議剔除) |
| pmts_overdue_1140A_overdue_rate__weighted_avg | TOP150入選-3 | numeric | 24.6873 | 高(建議剔除) |
| eir_270L | TOP150入選-2 | numeric | 22.9839 | 高(建議剔除) |
| interestrate_311L | TOP150入選-2 | numeric | 22.7085 | 高(建議剔除) |
| pmts_dpd_1073P_non_null_count__max | TOP150入選-3 | numeric | 20.5582 | 高(建議剔除) |
| pmts_dpd_303P_std__mean | TOP150入選-5 | numeric | 20.1270 | 高(建議剔除) |
| pmts_overdue_1152A_mean__mean | WOE | numeric | 19.8892 | 高(建議剔除) |
| pmts_dpd_303P_overdue_rate__weighted_avg | TOP150入選-3 | numeric | 18.1772 | 高(建議剔除) |
| pmts_overdue_1152A_overdue_rate__weighted_avg | TOP150入選-3 | numeric | 17.5513 | 高(建議剔除) |
| pctinstlsallpaidlate6d_3546844L | WOE | numeric | 16.3814 | 高(建議剔除) |
| pmts_dpd_1073P_non_null_count__sum | TOP150入選-1 | numeric | 16.2776 | 高(建議剔除) |
| b2_pmts_dpdvalue_108P_mean_weighted | AUC+KS | numeric | 15.3945 | 高(建議剔除) |
| registaddr_district_1083M_appl | TOP150入選-3 | categorical | 15.3689 | 高(建議剔除) |
| pmts_dpd_303P_std__max | WOE | numeric | 15.1037 | 高(建議剔除) |
| maxdpdlast12m_727P | TOP150入選-3 | numeric | 14.6214 | 高(建議剔除) |
| overdueamountmax2_398A__max | TOP150入選-1 | numeric | 13.9918 | 高(建議剔除) |
| contaddr_district_15M_appl | TOP150入選-2 | categorical | 13.7625 | 高(建議剔除) |
| b2_pmts_dpdvalue_108P_mean_contract_mean | AUC+KS | numeric | 13.6355 | 高(建議剔除) |
| pctinstlsallpaidlate4d_3546849L | TOP150入選-2 | numeric | 13.5864 | 高(建議剔除) |
| pmts_overdue_1140A_non_null_count__sum | EBM | numeric | 13.4557 | 高(建議剔除) |
| pmts_overdue_1140A_non_null_count__max | EBM | numeric | 13.3784 | 高(建議剔除) |
| n_rejected | TOP150入選-2 | numeric | 12.9064 | 高(建議剔除) |
| pmts_dpd_303P_recent3_mean__mean | TOP150入選-2 | numeric | 12.8397 | 高(建議剔除) |
| pmts_dpd_1073P_mean_positive__max | TOP150入選-1 | numeric | 12.4226 | 高(建議剔除) |
| totaldebt_9A | TOP150入選-2 | numeric | 11.4468 | 高(建議剔除) |
| datelastunpaid_3546854D | GBM | categorical | 11.3609 | 高(建議剔除) |
| pmts_overdue_1152A_std__mean | WOE | numeric | 11.2548 | 高(建議剔除) |
| numinsttopaygr_769L | TOP150入選-2 | numeric | 11.1523 | 高(建議剔除) |
| currdebt_22A | TOP150入選-2 | numeric | 11.0514 | 高(建議剔除) |
| numinstunpaidmax_3546851L | TOP150入選-2 | numeric | 11.0282 | 高(建議剔除) |
| pmts_dpd_1073P_longest_good_streak__max | TOP150入選-4 | numeric | 11.0060 | 高(建議剔除) |
| maxdpdlast9m_1059P | TOP150入選-3 | numeric | 10.7966 | 高(建議剔除) |
| maxdpdlast6m_474P | TOP150入選-2 | numeric | 10.5351 | 高(建議剔除) |
| days120_123L | TOP150入選-2 | numeric | 10.4564 | 高(建議剔除) |
| maxdpdtolerance_374P | TOP150入選-2 | numeric | 9.9104 | 中(留意) |
| pmts_overdue_1152A_median__mean | WOE | numeric | 9.5731 | 中(留意) |
| totalamount_996A__max | TOP150入選-1 | numeric | 9.5573 | 中(留意) |
| pctinstlsallpaidlat10d_839L | TOP150入選-2 | numeric | 9.0712 | 中(留意) |
| residualamount_856A__mean | TOP150入選-2 | numeric | 9.0374 | 中(留意) |
| totalamount_996A__mean | TOP150入選-2 | numeric | 8.5565 | 中(留意) |
| pmts_dpd_303P_median__mean | WOE | numeric | 8.3095 | 中(留意) |
| pmtamount_36A_sum_positive | TOP150入選-2 | numeric | 8.1056 | 中(留意) |
| pmtssum_45A | TOP150入選-3 | numeric | 7.7587 | 中(留意) |
| rejectreason_755M_non_placeholder_ratio | TOP150入選-2 | numeric | 7.6643 | 中(留意) |
| maxdpdinstldate_3546855D | GBM | categorical | 7.6227 | 中(留意) |
| pctinstlsallpaidlate1d_3546856L | TOP150入選-5 | numeric | 7.6188 | 中(留意) |
| days180_256L | TOP150入選-2 | numeric | 7.5246 | 中(留意) |
| days90_310L | TOP150入選-2 | numeric | 7.5099 | 中(留意) |
| rejectreasonclient_4145042M_non_placeholder_ratio | TOP150入選-4 | numeric | 7.4050 | 中(留意) |
| pmts_dpd_303P_mean_positive__recomputed | TOP150入選-2 | numeric | 7.4010 | 中(留意) |
| lastrejectdate_50D | GBM | categorical | 7.3255 | 中(留意) |
| pmts_dpd_1073P_trend__mean | TOP150入選-3 | numeric | 7.3077 | 中(留意) |
| maxdbddpdtollast12m_3658940P | TOP150入選-5 | numeric | 7.2564 | 中(留意) |
| prolongationcount_599L__null_rate | TOP150入選-2 | numeric | 7.1060 | 中(留意) |
| numinstlswithdpd10_728L | TOP150入選-2 | numeric | 7.0487 | 中(留意) |
| dpdmax_757P__null_rate | TOP150入選-1 | numeric | 6.6942 | 中(留意) |
| numinstlswithoutdpd_562L | TOP150入選-2 | numeric | 6.2860 | 中(留意) |
| pmts_overdue_1140A_n_unique__sum | EBM | numeric | 6.0522 | 中(留意) |
| lastdelinqdate_224D | GBM | categorical | 5.8302 | 中(留意) |
| reject_rate | TOP150入選-5 | numeric | 5.6853 | 中(留意) |
| monthsannuity_845L | TOP150入選-3 | numeric | 5.6752 | 中(留意) |
| maxdpdlast24m_143P | TOP150入選-3 | numeric | 5.4305 | 中(留意) |
| pmts_dpd_1073P_trend__max | TOP150入選-4 | numeric | 5.3298 | 中(留意) |
| totalamount_6A__null_rate | TOP150入選-1 | numeric | 5.2867 | 中(留意) |
| disbursedcredamount_1113A | TOP150入選-3 | numeric | 5.2696 | 中(留意) |
| outstandingamount_362A__max | TOP150入選-1 | numeric | 5.1135 | 中(留意) |
| maxdbddpdtollast6m_4187119P | TOP150入選-2 | numeric | 4.9960 | 低(可接受) |
| pmts_year_507T_pmts_month_706T_min__min | GBM | numeric | 4.9797 | 低(可接受) |
| pmts_dpd_303P_non_null_count__sum | TOP150入選-2 | numeric | 4.9303 | 低(可接受) |
| numinstlswithdpd5_4187116L | TOP150入選-2 | numeric | 4.7593 | 低(可接受) |
| opencred_647L | EBM | categorical | 4.6952 | 低(可接受) |
| credamount_770A | TOP150入選-3 | numeric | 4.6377 | 低(可接受) |
| subjectroles_name_838M_n_unique__sum | EBM | numeric | 4.5971 | 低(可接受) |
| pmts_dpd_303P_n_unique__sum | EBM | numeric | 4.5267 | 低(可接受) |
| maxdpdlast3m_392P | TOP150入選-2 | numeric | 4.3469 | 低(可接受) |
| pmts_dpd_303P_recent_time_key__mean_fallback | TOP150入選-2 | numeric | 4.1831 | 低(可接受) |
| totalsettled_863A | TOP150入選-2 | numeric | 4.0469 | 低(可接受) |
| overdueamountmax_155A__std | TOP150入選-1 | numeric | 3.9838 | 低(可接受) |
| cancelreason_3545846M_mode | TOP150入選-2 | categorical | 3.7738 | 低(可接受) |
| numberofoverdueinstlmax_1039L__min | TOP150入選-2 | numeric | 3.7645 | 低(可接受) |
| avgdpdtolclosure24_3658938P | TOP150入選-5 | numeric | 3.6390 | 低(可接受) |
| subjectroles_name_838M_entropy__mean | EBM | numeric | 3.6262 | 低(可接受) |
| totalamount_6A__sum | TOP150入選-2 | numeric | 3.3734 | 低(可接受) |
| pmts_year_1139T_pmts_month_158T_min__min | GBM | numeric | 3.3715 | 低(可接受) |
| lastrejectreason_759M | TOP150入選-3 | categorical | 3.2877 | 低(可接受) |
| maxdpdtolerance_577P_mean | TOP150入選-4 | numeric | 3.1117 | 低(可接受) |
| residualamount_856A__min | TOP150入選-2 | numeric | 3.0470 | 低(可接受) |
| collater_valueofguarantee_1124L_null_count__max | TOP150入選-2 | numeric | 2.9436 | 低(可接受) |
| maxdebt4_972A | TOP150入選-2 | numeric | 2.8726 | 低(可接受) |
| requesttype_4525192L | TOP150入選-3 | categorical | 2.8523 | 低(可接受) |
| avgdbddpdlast24m_3658932P | TOP150入選-3 | numeric | 2.8142 | 低(可接受) |
| totaloutstanddebtvalue_39A__sum | TOP150入選-1 | numeric | 2.7110 | 低(可接受) |
| pmts_dpd_303P_longest_good_streak__max | TOP150入選-4 | numeric | 2.6152 | 低(可接受) |
| numinstunpaidmaxest_4493212L | TOP150入選-2 | numeric | 2.5968 | 低(可接受) |
| pmts_dpd_303P_trend__mean | TOP150入選-3 | numeric | 2.4586 | 低(可接受) |
| last_status | Catboost | categorical | 2.4000 | 低(可接受) |
| pmts_year_1139T_pmts_month_158T_duration__mean | EBM | numeric | 2.2926 | 低(可接受) |
| pctinstlsallpaidearl3d_427L | TOP150入選-4 | numeric | 2.2468 | 低(可接受) |
| age_years_appl | TOP150入選-3 | numeric | 2.2448 | 低(可接受) |
| incometype_1044T_appl | TOP150入選-3 | categorical | 2.1920 | 低(可接受) |
| numrejects9m_859L | TOP150入選-3 | numeric | 2.1614 | 低(可接受) |
| days30_165L | TOP150入選-2 | numeric | 2.1241 | 低(可接受) |
| cntpmts24_3658933L | TOP150入選-3 | numeric | 2.1013 | 低(可接受) |
| pmts_dpd_1073P_recent_time_key__mean_fallback | EBM | numeric | 1.9891 | 低(可接受) |
| credtype_322L | Catboost | categorical | 1.9301 | 低(可接受) |
| price_1097A | TOP150入選-3 | numeric | 1.9258 | 低(可接受) |
| disbursementtype_67L | TOP150入選-2 | categorical | 1.9240 | 低(可接受) |
| registaddr_zipcode_184M_appl | GBM | categorical | 1.8177 | 低(可接受) |
| amount_4527230A_sum_positive | TOP150入選-2 | numeric | 1.8127 | 低(可接受) |
| empl_industry_691L_appl | Catboost | categorical | 1.7508 | 低(可接受) |
| education_1138M_mode | TOP150入選-2 | categorical | 1.7440 | 低(可接受) |
| maxdbddpdlast1m_3658939P | TOP150入選-2 | numeric | 1.6995 | 低(可接受) |
| tenure_years_max | TOP150入選-3 | numeric | 1.6803 | 低(可接受) |
| numberofoutstandinstls_59L__min | TOP150入選-2 | numeric | 1.4349 | 低(可接受) |
| pmtaverage_3A | TOP150入選-3 | numeric | 1.4161 | 低(可接受) |
| tenure_years_appl | TOP150入選-3 | numeric | 1.3922 | 低(可接受) |
| numberofoutstandinstls_59L__sum | TOP150入選-2 | numeric | 1.3800 | 低(可接受) |
| riskassesment_940T | AUC+KS | numeric | 1.2948 | 低(可接受) |
| familystate_447L_appl | TOP150入選-2 | categorical | 1.2332 | 低(可接受) |
| pmts_dpd_303P_recent_time_key__max_fallback | EBM | numeric | 1.1924 | 低(可接受) |
| education_927M_appl | TOP150入選-3 | categorical | 1.1677 | 低(可接受) |
| education_1103M | TOP150入選-3 | categorical | 1.1344 | 低(可接受) |
| monthlyinstlamount_674A__mean | TOP150入選-1 | numeric | 1.0572 | 低(可接受) |
| maxannuity_4075009A | Catboost | numeric | 1.0082 | 低(可接受) |

## 高 VIF 欄位(> 10,建議人工複查是否剔除)
| column | source | VIF |
|---|---|---|
| pmtamount_36A_positive_count | TOP150入選-2 | 155029.8311 |
| pmtamount_36A_non_null_count | Catboost | 38756.7492 |
| tax_registry_c_count | TOP150入選-2 | 38756.7492 |
| amount_4527230A_positive_count | Catboost | 27493.0546 |
| tax_registry_a_count | TOP150入選-2 | 27487.7593 |
| relationshiptoclient_415T_mode | TOP150入選-2 | 2036.9975 |
| relationshiptoclient_642T_mode | TOP150入選-2 | 2036.9198 |
| b2_pmts_pmtsoverdue_635A_overdue_rate_recomputed | AUC+KS | 888.1976 |
| b2_pmts_dpdvalue_108P_overdue_rate_recomputed | AUC+KS | 887.5829 |
| days360_512L | TOP150入選-2 | 158.6245 |
| numberofqueries_373L | Catboost | 156.8122 |
| dpdmax_139P__max | TOP150入選-1 | 142.5017 |
| prolongationcount_599L__sum | TOP150入選-1 | 138.1013 |
| prolongationcount_599L__mean | GBM | 127.8055 |
| pmts_dpd_1073P_positive_count__max | TOP150入選-1 | 119.5198 |
| pmts_dpd_1073P_sum_positive__max | TOP150入選-1 | 115.0709 |
| dpdmax_139P__sum | TOP150入選-3 | 112.8040 |
| pmts_dpd_1073P_positive_count__sum | TOP150入選-1 | 104.9287 |
| pmts_dpd_1073P_sum_positive__sum | TOP150入選-1 | 102.3072 |
| overdueamountmax_155A__sum | TOP150入選-2 | 101.6908 |
| pmts_overdue_1140A_sum_positive__max | TOP150入選-2 | 100.7600 |
| overdueamountmax2_14A__sum | TOP150入選-1 | 97.6194 |
| pmts_overdue_1140A_sum_positive__sum | TOP150入選-3 | 91.9105 |
| overdueamountmax_155A__mean | TOP150入選-4 | 75.1624 |
| numberofoverdueinstlmax_1039L__sum | TOP150入選-3 | 72.6657 |
| overdueamountmax2_14A__max | TOP150入選-2 | 72.3234 |
| numberofoverdueinstlmax_1039L__max | TOP150入選-3 | 68.9810 |
| overdueamountmax_155A__max | TOP150入選-2 | 68.1557 |
| numberofoverdueinstlmax_1151L__max | TOP150入選-2 | 64.8706 |
| dpdmax_757P__max | TOP150入選-2 | 63.8162 |
| numberofoverdueinstlmax_1151L__sum | TOP150入選-2 | 61.5314 |
| pmts_overdue_1140A_mean__max | WOE | 61.5306 |
| pmts_dpd_1073P_mean__mean | TOP150入選-1 | 59.7128 |
| pmts_overdue_1140A_std__max | TOP150入選-2 | 59.4908 |
| dpdmax_139P__mean | TOP150入選-4 | 58.7303 |
| dpdmax_757P__sum | TOP150入選-2 | 56.8406 |
| pmts_dpd_1073P_mean__max | TOP150入選-1 | 56.3269 |
| pmts_dpd_1073P_max__max | TOP150入選-1 | 54.6489 |
| pmts_overdue_1140A_std__mean | TOP150入選-3 | 54.1177 |
| pmts_overdue_1140A_mean__mean | TOP150入選-4 | 52.1347 |
| b2_pmts_dpdvalue_108P_overdue_rate_contract_max | AUC+KS | 49.3453 |
| b2_pmts_pmtsoverdue_635A_overdue_rate_contract_max | AUC+KS | 48.2880 |
| pmts_dpd_303P_recent12_mean__max | WOE | 47.4555 |
| pmts_dpd_1073P_std__mean | TOP150入選-4 | 47.1172 |
| pmts_dpd_1073P_recent12_mean__max | WOE | 45.0530 |
| overdueamountmax2_14A__mean | TOP150入選-2 | 44.9873 |
| pmts_dpd_1073P_recent12_mean__mean | WOE | 42.3986 |
| numberofoverdueinstlmax_1039L__mean | TOP150入選-4 | 40.1148 |
| pmts_dpd_303P_mean__max | TOP150入選-1 | 39.8531 |
| pmts_dpd_1073P_n_unique__max | TOP150入選-1 | 38.3925 |
| overdueamountmax_35A__mean | TOP150入選-2 | 38.0957 |
| pmts_dpd_303P_max__max | TOP150入選-1 | 37.3312 |
| pmts_dpd_303P_mean__mean | TOP150入選-3 | 36.3870 |
| pmts_dpd_303P_recent12_mean__mean | TOP150入選-2 | 35.7230 |
| pmts_dpd_303P_recent6_mean__max | TOP150入選-2 | 34.2103 |
| pmts_dpd_1073P_overdue_rate__weighted_avg | TOP150入選-3 | 33.5941 |
| b2_pmts_pmtsoverdue_635A_mean_contract_max | AUC+KS | 33.1214 |
| residualamount_856A__max | TOP150入選-2 | 32.6611 |
| numberofoverdueinstlmax_1151L__std | TOP150入選-3 | 32.3967 |
| pmts_dpd_303P_recent6_mean__mean | TOP150入選-2 | 32.2720 |
| b2_pmts_pmtsoverdue_635A_mean_weighted | AUC+KS | 31.9293 |
| b2_pmts_pmtsoverdue_635A_mean_contract_mean | AUC+KS | 31.2208 |
| pmts_dpd_1073P_std__max | TOP150入選-2 | 30.2417 |
| dpdmax_757P__std | TOP150入選-4 | 30.1649 |
| residualamount_856A__sum | TOP150入選-2 | 29.4215 |
| overdueamountmax2_398A__mean | TOP150入選-1 | 28.9306 |
| overdueamountmax2_398A__std | TOP150入選-1 | 28.2170 |
| numberofoverdueinstlmax_1151L__mean | TOP150入選-4 | 27.9832 |
| dpdmax_757P__mean | TOP150入選-4 | 27.7051 |
| pmts_dpd_1073P_consecutive_max__max | TOP150入選-2 | 27.4478 |
| overdueamountmax_35A__std | TOP150入選-2 | 25.1011 |
| pmts_overdue_1140A_overdue_rate__weighted_avg | TOP150入選-3 | 24.6873 |
| eir_270L | TOP150入選-2 | 22.9839 |
| interestrate_311L | TOP150入選-2 | 22.7085 |
| pmts_dpd_1073P_non_null_count__max | TOP150入選-3 | 20.5582 |
| pmts_dpd_303P_std__mean | TOP150入選-5 | 20.1270 |
| pmts_overdue_1152A_mean__mean | WOE | 19.8892 |
| pmts_dpd_303P_overdue_rate__weighted_avg | TOP150入選-3 | 18.1772 |
| pmts_overdue_1152A_overdue_rate__weighted_avg | TOP150入選-3 | 17.5513 |
| pctinstlsallpaidlate6d_3546844L | WOE | 16.3814 |
| pmts_dpd_1073P_non_null_count__sum | TOP150入選-1 | 16.2776 |
| b2_pmts_dpdvalue_108P_mean_weighted | AUC+KS | 15.3945 |
| registaddr_district_1083M_appl | TOP150入選-3 | 15.3689 |
| pmts_dpd_303P_std__max | WOE | 15.1037 |
| maxdpdlast12m_727P | TOP150入選-3 | 14.6214 |
| overdueamountmax2_398A__max | TOP150入選-1 | 13.9918 |
| contaddr_district_15M_appl | TOP150入選-2 | 13.7625 |
| b2_pmts_dpdvalue_108P_mean_contract_mean | AUC+KS | 13.6355 |
| pctinstlsallpaidlate4d_3546849L | TOP150入選-2 | 13.5864 |
| pmts_overdue_1140A_non_null_count__sum | EBM | 13.4557 |
| pmts_overdue_1140A_non_null_count__max | EBM | 13.3784 |
| n_rejected | TOP150入選-2 | 12.9064 |
| pmts_dpd_303P_recent3_mean__mean | TOP150入選-2 | 12.8397 |
| pmts_dpd_1073P_mean_positive__max | TOP150入選-1 | 12.4226 |
| totaldebt_9A | TOP150入選-2 | 11.4468 |
| datelastunpaid_3546854D | GBM | 11.3609 |
| pmts_overdue_1152A_std__mean | WOE | 11.2548 |
| numinsttopaygr_769L | TOP150入選-2 | 11.1523 |
| currdebt_22A | TOP150入選-2 | 11.0514 |
| numinstunpaidmax_3546851L | TOP150入選-2 | 11.0282 |
| pmts_dpd_1073P_longest_good_streak__max | TOP150入選-4 | 11.0060 |
| maxdpdlast9m_1059P | TOP150入選-3 | 10.7966 |
| maxdpdlast6m_474P | TOP150入選-2 | 10.5351 |
| days120_123L | TOP150入選-2 | 10.4564 |

完整結果見 `data/final_feature_vif.csv`。
