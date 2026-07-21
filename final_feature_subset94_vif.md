# data/final_feature.csv 候選特徵 VIF 多重共線性報告
## Context
團隊以多種方法(AUC+KS、Catboost、EBM、GBM、TOP150 入選-1~5、WOE)彙整出 `data/final_feature.csv` 195 個候選特徵。本報告排除非特徵中繼欄 `date_decision` 後,對其餘 94 個候選特徵計算 VIF (Variance Inflation Factor),檢查多重共線性,供後續人工決定去留(本報告為 report-only,不自動剔除欄位)。
**本次為自訂子集重算**:從 194 個候選特徵中選取 94 個欄位重新計算 VIF(欄位清單來源:`C:/Users/ITTRAI~1/AppData/Local/Temp/claude/c--Users-ittraining-Documents-Default-risk-ML/7a2af10b-ff29-41d5-b1a3-8380c1552a65/scratchpad/vif_recalc_columns.txt`)。
## 方法
1. 以 DuckDB reservoir sampling(固定 seed=42)從 `data/base_final.parquet` 取出 500,000 列,所有候選欄共用同一批列以保留欄位間相關結構。
2. 每欄以與 `feature_selection_woe_iv.py` 相同的決策樹分箱(數值/時間戳)或類別稀有併箱(類別欄)+ `category_encoders.WOEEncoder` 轉換為 WOE 數值欄,組成 500,000 × 94 的 WOE 矩陣。
3. VIF 以相關矩陣虛擬逆(`numpy.linalg.pinv`)的對角線計算(與加入常數項的逐欄 OLS 迴歸法數學等價,但快得多);另抽樣欄位以 `statsmodels.variance_inflation_factor` 獨立重算比對一致性(見下方驗證章節)。
## 分層門檻
| VIF 區間 | 層級 |
|---|---|
| < 5 | 低(可接受) |
| 5 – 10 | 中(留意) |
| > 10 | 高(建議剔除) |

## 結果摘要
- 分析欄位總數:94

| 層級 | 欄位數 |
|---|---|
| 低(可接受) | 75 |
| 中(留意) | 12 |
| 高(建議剔除) | 7 |

## 驗證:statsmodels 交叉比對
| column | corr-inverse VIF | statsmodels VIF |
|---|---|---|
| overdueamountmax_155A__mean | 12.4822 | 12.4822 |
| maxdpdlast24m_143P | 4.5827 | 4.5827 |
| lastrejectreason_759M | 2.9593 | 2.9593 |
| credtype_322L | 1.8799 | 1.8799 |
| maxannuity_4075009A | 1.0076 | 1.0076 |

## 完整 VIF 排名
| column | source | var_type | VIF | tier |
|---|---|---|---|---|
| overdueamountmax_155A__mean | TOP150入選-4 | numeric | 12.4822 | 高(建議剔除) |
| pmts_overdue_1140A_std__max | TOP150入選-2 | numeric | 11.9937 | 高(建議剔除) |
| pmts_dpd_303P_std__mean | TOP150入選-5 | numeric | 11.6497 | 高(建議剔除) |
| dpdmax_139P__mean | TOP150入選-4 | numeric | 11.6391 | 高(建議剔除) |
| pmts_dpd_1073P_std__mean | TOP150入選-4 | numeric | 11.2615 | 高(建議剔除) |
| dpdmax_757P__sum | TOP150入選-2 | numeric | 10.3878 | 高(建議剔除) |
| b2_pmts_pmtsoverdue_635A_mean_weighted | AUC+KS | numeric | 10.2110 | 高(建議剔除) |
| b2_pmts_dpdvalue_108P_overdue_rate_recomputed | AUC+KS | numeric | 9.7240 | 中(留意) |
| pmts_dpd_303P_std__max | WOE | numeric | 9.5995 | 中(留意) |
| maxdpdtolerance_374P | TOP150入選-2 | numeric | 9.0422 | 中(留意) |
| b2_pmts_dpdvalue_108P_overdue_rate_contract_max | AUC+KS | numeric | 8.6572 | 中(留意) |
| maxdpdlast12m_727P | TOP150入選-3 | numeric | 7.3601 | 中(留意) |
| pmts_dpd_1073P_trend__mean | TOP150入選-3 | numeric | 6.6058 | 中(留意) |
| b2_pmts_dpdvalue_108P_mean_weighted | AUC+KS | numeric | 6.2060 | 中(留意) |
| numinstlswithdpd10_728L | TOP150入選-2 | numeric | 6.1224 | 中(留意) |
| pmts_overdue_1140A_overdue_rate__weighted_avg | TOP150入選-3 | numeric | 5.6647 | 中(留意) |
| days180_256L | TOP150入選-2 | numeric | 5.5893 | 中(留意) |
| overdueamountmax_35A__mean | TOP150入選-2 | numeric | 5.4464 | 中(留意) |
| rejectreasonclient_4145042M_non_placeholder_ratio | TOP150入選-4 | numeric | 5.2160 | 中(留意) |
| pmts_year_507T_pmts_month_706T_min__min | GBM | numeric | 4.9770 | 低(可接受) |
| pmts_dpd_1073P_trend__max | TOP150入選-4 | numeric | 4.8180 | 低(可接受) |
| maxdpdinstldate_3546855D | GBM | categorical | 4.7648 | 低(可接受) |
| dpdmax_757P__null_rate | TOP150入選-1 | numeric | 4.6258 | 低(可接受) |
| maxdpdlast24m_143P | TOP150入選-3 | numeric | 4.5827 | 低(可接受) |
| opencred_647L | EBM | categorical | 4.5724 | 低(可接受) |
| pmts_dpd_303P_mean_positive__recomputed | TOP150入選-2 | numeric | 4.5044 | 低(可接受) |
| maxdbddpdtollast12m_3658940P | TOP150入選-5 | numeric | 4.4689 | 低(可接受) |
| pmts_dpd_303P_non_null_count__sum | TOP150入選-2 | numeric | 4.3893 | 低(可接受) |
| pctinstlsallpaidlate1d_3546856L | TOP150入選-5 | numeric | 4.3640 | 低(可接受) |
| days90_310L | TOP150入選-2 | numeric | 4.2712 | 低(可接受) |
| numinstlswithoutdpd_562L | TOP150入選-2 | numeric | 4.2123 | 低(可接受) |
| pmts_dpd_303P_recent_time_key__mean_fallback | TOP150入選-2 | numeric | 4.1942 | 低(可接受) |
| numinstlswithdpd5_4187116L | TOP150入選-2 | numeric | 3.9950 | 低(可接受) |
| days360_512L | TOP150入選-2 | numeric | 3.7791 | 低(可接受) |
| lastdelinqdate_224D | GBM | categorical | 3.7191 | 低(可接受) |
| lastrejectdate_50D | GBM | categorical | 3.6140 | 低(可接受) |
| overdueamountmax_35A__std | TOP150入選-2 | numeric | 3.5401 | 低(可接受) |
| pmts_dpd_303P_n_unique__sum | EBM | numeric | 3.5349 | 低(可接受) |
| avgdpdtolclosure24_3658938P | TOP150入選-5 | numeric | 3.5152 | 低(可接受) |
| totalsettled_863A | TOP150入選-2 | numeric | 3.5138 | 低(可接受) |
| pmts_overdue_1140A_n_unique__sum | EBM | numeric | 3.4630 | 低(可接受) |
| subjectroles_name_838M_n_unique__sum | EBM | numeric | 3.4442 | 低(可接受) |
| cancelreason_3545846M_mode | TOP150入選-2 | categorical | 3.3685 | 低(可接受) |
| outstandingamount_362A__max | TOP150入選-1 | numeric | 3.3483 | 低(可接受) |
| totalamount_996A__mean | TOP150入選-2 | numeric | 3.2073 | 低(可接受) |
| reject_rate | TOP150入選-5 | numeric | 3.1078 | 低(可接受) |
| lastrejectreason_759M | TOP150入選-3 | categorical | 2.9593 | 低(可接受) |
| totalamount_6A__sum | TOP150入選-2 | numeric | 2.8838 | 低(可接受) |
| maxdpdtolerance_577P_mean | TOP150入選-4 | numeric | 2.8699 | 低(可接受) |
| overdueamountmax_155A__std | TOP150入選-1 | numeric | 2.8159 | 低(可接受) |
| maxdebt4_972A | TOP150入選-2 | numeric | 2.7605 | 低(可接受) |
| pmts_year_1139T_pmts_month_158T_min__min | GBM | numeric | 2.7205 | 低(可接受) |
| requesttype_4525192L | TOP150入選-3 | categorical | 2.6274 | 低(可接受) |
| pmts_overdue_1140A_non_null_count__max | EBM | numeric | 2.5866 | 低(可接受) |
| subjectroles_name_838M_entropy__mean | EBM | numeric | 2.5221 | 低(可接受) |
| pmts_dpd_303P_trend__mean | TOP150入選-3 | numeric | 2.3976 | 低(可接受) |
| maxdpdlast3m_392P | TOP150入選-2 | numeric | 2.3332 | 低(可接受) |
| last_status | Catboost | categorical | 2.3180 | 低(可接受) |
| pmts_dpd_303P_longest_good_streak__max | TOP150入選-4 | numeric | 2.2731 | 低(可接受) |
| age_years_appl | TOP150入選-3 | numeric | 2.1880 | 低(可接受) |
| incometype_1044T_appl | TOP150入選-3 | categorical | 2.1220 | 低(可接受) |
| collater_valueofguarantee_1124L_null_count__max | TOP150入選-2 | numeric | 2.1146 | 低(可接受) |
| days30_165L | TOP150入選-2 | numeric | 2.1130 | 低(可接受) |
| numberofoverdueinstlmax_1039L__min | TOP150入選-2 | numeric | 2.1123 | 低(可接受) |
| pmts_year_1139T_pmts_month_158T_duration__mean | EBM | numeric | 2.0938 | 低(可接受) |
| amount_4527230A_positive_count | Catboost | numeric | 2.0799 | 低(可接受) |
| cntpmts24_3658933L | TOP150入選-3 | numeric | 2.0339 | 低(可接受) |
| numrejects9m_859L | TOP150入選-3 | numeric | 2.0002 | 低(可接受) |
| pctinstlsallpaidearl3d_427L | TOP150入選-4 | numeric | 1.8966 | 低(可接受) |
| credtype_322L | Catboost | categorical | 1.8799 | 低(可接受) |
| disbursedcredamount_1113A | TOP150入選-3 | numeric | 1.8286 | 低(可接受) |
| pmts_dpd_1073P_recent_time_key__mean_fallback | EBM | numeric | 1.8191 | 低(可接受) |
| price_1097A | TOP150入選-3 | numeric | 1.7895 | 低(可接受) |
| amount_4527230A_sum_positive | TOP150入選-2 | numeric | 1.7822 | 低(可接受) |
| disbursementtype_67L | TOP150入選-2 | categorical | 1.7198 | 低(可接受) |
| empl_industry_691L_appl | Catboost | categorical | 1.7169 | 低(可接受) |
| education_1138M_mode | TOP150入選-2 | categorical | 1.7163 | 低(可接受) |
| tenure_years_max | TOP150入選-3 | numeric | 1.6634 | 低(可接受) |
| numinstunpaidmaxest_4493212L | TOP150入選-2 | numeric | 1.4736 | 低(可接受) |
| tenure_years_appl | TOP150入選-3 | numeric | 1.3799 | 低(可接受) |
| pmtaverage_3A | TOP150入選-3 | numeric | 1.3782 | 低(可接受) |
| numberofoutstandinstls_59L__sum | TOP150入選-2 | numeric | 1.3719 | 低(可接受) |
| numberofoutstandinstls_59L__min | TOP150入選-2 | numeric | 1.3708 | 低(可接受) |
| riskassesment_940T | AUC+KS | numeric | 1.2815 | 低(可接受) |
| residualamount_856A__min | TOP150入選-2 | numeric | 1.2375 | 低(可接受) |
| familystate_447L_appl | TOP150入選-2 | categorical | 1.1885 | 低(可接受) |
| pmts_dpd_303P_recent_time_key__max_fallback | EBM | numeric | 1.1808 | 低(可接受) |
| education_927M_appl | TOP150入選-3 | categorical | 1.1572 | 低(可接受) |
| education_1103M | TOP150入選-3 | categorical | 1.1221 | 低(可接受) |
| prolongationcount_599L__null_rate | TOP150入選-2 | numeric | 1.1191 | 低(可接受) |
| interestrate_311L | TOP150入選-2 | numeric | 1.1157 | 低(可接受) |
| registaddr_zipcode_184M_appl | GBM | categorical | 1.0728 | 低(可接受) |
| monthlyinstlamount_674A__mean | TOP150入選-1 | numeric | 1.0591 | 低(可接受) |
| maxannuity_4075009A | Catboost | numeric | 1.0076 | 低(可接受) |

## 高 VIF 欄位(> 10,建議人工複查是否剔除)
| column | source | VIF |
|---|---|---|
| overdueamountmax_155A__mean | TOP150入選-4 | 12.4822 |
| pmts_overdue_1140A_std__max | TOP150入選-2 | 11.9937 |
| pmts_dpd_303P_std__mean | TOP150入選-5 | 11.6497 |
| dpdmax_139P__mean | TOP150入選-4 | 11.6391 |
| pmts_dpd_1073P_std__mean | TOP150入選-4 | 11.2615 |
| dpdmax_757P__sum | TOP150入選-2 | 10.3878 |
| b2_pmts_pmtsoverdue_635A_mean_weighted | AUC+KS | 10.2110 |

完整結果見 `data/final_feature_subset94_vif.csv`。
