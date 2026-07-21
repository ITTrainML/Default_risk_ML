# base_final.parquet WOE/IV 單變量特徵篩選報告
## Context
對 `data/base_final.parquet`(`data/base.parquet` LEFT JOIN `data/train_person_agg.parquet`,以 `case_id` 為鍵)中除 `case_id`、`date_decision`、`MONTH`、`WEEK_NUM`、`target` 以外的每個自變數,計算其對 `target`(1=違約、0=正常還款)的 Weight of Evidence (WOE) 與 Information Value (IV),以 IV 作為特徵去留依據。
## 方法
1. **分箱**:數值/時間戳欄以決策樹(`max_leaf_nodes=8`、`min_samples_leaf>=1000`、`class_weight=balanced`,對非空值擬合、抽樣上限 30 萬列,以控制執行時間)分箱;缺失值獨立成 `__MISSING__` 箱。類別欄以原值為箱,出現次數 < 1000 的稀有類別併入 `__RARE__`。
2. **WOE**:對分箱後的離散欄以 `category_encoders.WOEEncoder`(`regularization=0.5`)轉換,取得每箱 WOE。
3. **IV**:以各箱 good/bad 分布(套用與 WOEEncoder 相同的 regularization)自算 `IV = Σ(dist_bad - dist_good) × WOE`。
## 篩選門檻(Siddiqi 慣例)
| IV 區間 | 層級 | decision |
|---|---|---|
| < 0.02 | 無預測力 | drop |
| 0.02 – 0.10 | 弱 | keep |
| 0.10 – 0.30 | 中 | keep |
| 0.30 – 0.50 | 強 | keep |
| > 0.50 | 可疑(疑洩漏/過強,建議人工複查) | keep |

## 結果摘要
- 受篩欄位總數:1196
- keep:659;drop:537

| 層級 | 欄位數 |
|---|---|
| 無預測力 | 537 |
| 弱 | 473 |
| 中 | 177 |
| 強 | 9 |

## Top 40 IV 排名
| column | var_type | n_bins | missing_rate | IV | tier | decision | bin_intervals |
|---|---|---|---|---|---|---|---|
| pmts_overdue_1152A_overdue_rate__weighted_avg | numeric | 9 | 0.336863 | 0.357277 | 強 | keep | bin_0:(-inf, 0.0005886] | bin_1:(0.0005886, 0.06518] | bin_2:(0.06518, 0.1199] | bin_3:(0.1199, 0.1997] | bin_4:(0.1997, 0.2962] | bin_5:(0.2962, 0.3974] | bin_6:(0.3974, 0.6582] | bin_7:(0.6582, inf) | __MISSING__ |
| pmts_dpd_303P_overdue_rate__weighted_avg | numeric | 9 | 0.337875 | 0.351267 | 強 | keep | bin_0:(-inf, 0.002792] | bin_1:(0.002792, 0.01386] | bin_2:(0.01386, 0.06183] | bin_3:(0.06183, 0.1417] | bin_4:(0.1417, 0.2485] | bin_5:(0.2485, 0.363] | bin_6:(0.363, 0.6261] | bin_7:(0.6261, inf) | __MISSING__ |
| avgdpdtolclosure24_3658938P | numeric | 8 | 0.306011 | 0.319773 | 強 | keep | bin_0:(-inf, 0.5] | bin_1:(0.5, 1.5] | bin_2:(1.5, 2.5] | bin_3:(2.5, 5.5] | bin_4:(5.5, 11.5] | bin_5:(11.5, 210.5] | bin_6:(210.5, inf) | __MISSING__ |
| pmts_dpd_303P_std__mean | numeric | 9 | 0.341315 | 0.312981 | 強 | keep | bin_0:(-inf, 0.546] | bin_1:(0.546, 0.9414] | bin_2:(0.9414, 1.961] | bin_3:(1.961, 2.571] | bin_4:(2.571, 4.045] | bin_5:(4.045, 46.47] | bin_6:(46.47, 77.81] | bin_7:(77.81, inf) | __MISSING__ |
| pmts_dpd_303P_mean__mean | numeric | 9 | 0.337875 | 0.309476 | 強 | keep | bin_0:(-inf, 0.00211] | bin_1:(0.00211, 0.0714] | bin_2:(0.0714, 0.3325] | bin_3:(0.3325, 1.116] | bin_4:(1.116, 1.893] | bin_5:(1.893, 4.507] | bin_6:(4.507, 142.3] | bin_7:(142.3, inf) | __MISSING__ |
| pmts_overdue_1152A_mean__mean | numeric | 9 | 0.336863 | 0.30805 | 強 | keep | bin_0:(-inf, 2.674e-06] | bin_1:(2.674e-06, 70.84] | bin_2:(70.84, 190] | bin_3:(190, 283] | bin_4:(283, 605.2] | bin_5:(605.2, 1107] | bin_6:(1107, 5612] | bin_7:(5612, inf) | __MISSING__ |
| dpdmax_757P__mean | numeric | 9 | 0.336736 | 0.307872 | 強 | keep | bin_0:(-inf, 2.47] | bin_1:(2.47, 4.981] | bin_2:(4.981, 7.493] | bin_3:(7.493, 13.44] | bin_4:(13.44, 26.48] | bin_5:(26.48, 85.49] | bin_6:(85.49, 409.4] | bin_7:(409.4, inf) | __MISSING__ |
| numberofoverdueinstlmax_1151L__mean | numeric | 9 | 0.335759 | 0.303415 | 強 | keep | bin_0:(-inf, 2.915] | bin_1:(2.915, 6.282] | bin_2:(6.282, 9.829] | bin_3:(9.829, 17.06] | bin_4:(17.06, 143.6] | bin_5:(143.6, 252.9] | bin_6:(252.9, 474.9] | bin_7:(474.9, inf) | __MISSING__ |
| pmts_dpd_303P_recent12_mean__mean | numeric | 9 | 0.337875 | 0.300464 | 強 | keep | bin_0:(-inf, 0.002348] | bin_1:(0.002348, 0.1948] | bin_2:(0.1948, 0.4874] | bin_3:(0.4874, 1.034] | bin_4:(1.034, 1.581] | bin_5:(1.581, 5.055] | bin_6:(5.055, 178] | bin_7:(178, inf) | __MISSING__ |
| pmts_overdue_1140A_overdue_rate__weighted_avg | numeric | 9 | 0.173909 | 0.299811 | 中 | keep | bin_0:(-inf, 0.00939] | bin_1:(0.00939, 0.03117] | bin_2:(0.03117, 0.05759] | bin_3:(0.05759, 0.1084] | bin_4:(0.1084, 0.1721] | bin_5:(0.1721, 0.2432] | bin_6:(0.2432, 0.3395] | bin_7:(0.3395, inf) | __MISSING__ |
| pmts_dpd_303P_recent6_mean__mean | numeric | 9 | 0.337875 | 0.291139 | 中 | keep | bin_0:(-inf, 0.1578] | bin_1:(0.1578, 0.365] | bin_2:(0.365, 1.132] | bin_3:(1.132, 2.232] | bin_4:(2.232, 5.025] | bin_5:(5.025, 159.8] | bin_6:(159.8, 315.4] | bin_7:(315.4, inf) | __MISSING__ |
| pmts_dpd_1073P_overdue_rate__weighted_avg | numeric | 9 | 0.175446 | 0.290259 | 中 | keep | bin_0:(-inf, 0.03069] | bin_1:(0.03069, 0.04435] | bin_2:(0.04435, 0.06639] | bin_3:(0.06639, 0.1081] | bin_4:(0.1081, 0.1315] | bin_5:(0.1315, 0.205] | bin_6:(0.205, 0.34] | bin_7:(0.34, inf) | __MISSING__ |
| pmts_dpd_1073P_std__mean | numeric | 9 | 0.183774 | 0.287967 | 中 | keep | bin_0:(-inf, 0.053] | bin_1:(0.053, 0.1053] | bin_2:(0.1053, 0.2377] | bin_3:(0.2377, 0.8677] | bin_4:(0.8677, 2.823] | bin_5:(2.823, 4.836] | bin_6:(4.836, 10.09] | bin_7:(10.09, inf) | __MISSING__ |
| pctinstlsallpaidlate1d_3546856L | numeric | 9 | 0.300485 | 0.286725 | 中 | keep | bin_0:(-inf, 0.06919] | bin_1:(0.06919, 0.09644] | bin_2:(0.09644, 0.1415] | bin_3:(0.1415, 0.2451] | bin_4:(0.2451, 0.351] | bin_5:(0.351, 0.4695] | bin_6:(0.4695, 0.6959] | bin_7:(0.6959, inf) | __MISSING__ |
| dpdmax_139P__mean | numeric | 9 | 0.17238 | 0.280781 | 中 | keep | bin_0:(-inf, 0.1339] | bin_1:(0.1339, 1.586] | bin_2:(1.586, 2.586] | bin_3:(2.586, 5.071] | bin_4:(5.071, 8.225] | bin_5:(8.225, 13.59] | bin_6:(13.59, 30.9] | bin_7:(30.9, inf) | __MISSING__ |
| pmts_overdue_1152A_std__mean | numeric | 9 | 0.339997 | 0.280651 | 中 | keep | bin_0:(-inf, 119.6] | bin_1:(119.6, 264.7] | bin_2:(264.7, 469.2] | bin_3:(469.2, 583.6] | bin_4:(583.6, 874.6] | bin_5:(874.6, 1577] | bin_6:(1577, 3000] | bin_7:(3000, inf) | __MISSING__ |
| pmts_overdue_1140A_mean__mean | numeric | 9 | 0.173909 | 0.279425 | 中 | keep | bin_0:(-inf, 18.08] | bin_1:(18.08, 42.83] | bin_2:(42.83, 94.9] | bin_3:(94.9, 205.1] | bin_4:(205.1, 293] | bin_5:(293, 574.6] | bin_6:(574.6, 1388] | bin_7:(1388, inf) | __MISSING__ |
| pmts_dpd_303P_recent3_mean__mean | numeric | 9 | 0.337875 | 0.279001 | 中 | keep | bin_0:(-inf, 0.3143] | bin_1:(0.3143, 0.9787] | bin_2:(0.9787, 2.138] | bin_3:(2.138, 4.66] | bin_4:(4.66, 16.95] | bin_5:(16.95, 164.5] | bin_6:(164.5, 325.9] | bin_7:(325.9, inf) | __MISSING__ |
| pmts_dpd_1073P_mean__mean | numeric | 9 | 0.175446 | 0.275784 | 中 | keep | bin_0:(-inf, 0.02836] | bin_1:(0.02836, 0.07986] | bin_2:(0.07986, 0.2255] | bin_3:(0.2255, 0.4875] | bin_4:(0.4875, 1.235] | bin_5:(1.235, 2.688] | bin_6:(2.688, 6.214] | bin_7:(6.214, inf) | __MISSING__ |
| overdueamountmax_35A__mean | numeric | 9 | 0.336674 | 0.275377 | 中 | keep | bin_0:(-inf, 374.5] | bin_1:(374.5, 664] | bin_2:(664, 991.8] | bin_3:(991.8, 1371] | bin_4:(1371, 2256] | bin_5:(2256, 4211] | bin_6:(4211, 1.516e+04] | bin_7:(1.516e+04, inf) | __MISSING__ |
| maxdpdtolerance_577P_mean | numeric | 9 | 0.199872 | 0.2747 | 中 | keep | bin_0:(-inf, 0.02941] | bin_1:(0.02941, 0.1938] | bin_2:(0.1938, 0.4833] | bin_3:(0.4833, 1.134] | bin_4:(1.134, 2.481] | bin_5:(2.481, 3.652] | bin_6:(3.652, 12.95] | bin_7:(12.95, inf) | __MISSING__ |
| dpdmax_139P__sum | numeric | 9 | 0.17238 | 0.272878 | 中 | keep | bin_0:(-inf, 0.5] | bin_1:(0.5, 1.5] | bin_2:(1.5, 4.5] | bin_3:(4.5, 10.5] | bin_4:(10.5, 22.5] | bin_5:(22.5, 35.5] | bin_6:(35.5, 114.5] | bin_7:(114.5, inf) | __MISSING__ |
| pmts_dpd_1073P_trend__mean | numeric | 9 | 0.175446 | 0.271646 | 中 | keep | bin_0:(-inf, -0.1434] | bin_1:(-0.1434, -0.01499] | bin_2:(-0.01499, -0.002591] | bin_3:(-0.002591, 0.003094] | bin_4:(0.003094, 0.03369] | bin_5:(0.03369, 0.1361] | bin_6:(0.1361, 1.477] | bin_7:(1.477, inf) | __MISSING__ |
| overdueamountmax2_398A__mean | numeric | 9 | 0.335759 | 0.270457 | 中 | keep | bin_0:(-inf, 0.008711] | bin_1:(0.008711, 441.9] | bin_2:(441.9, 692.6] | bin_3:(692.6, 1635] | bin_4:(1635, 2676] | bin_5:(2676, 5852] | bin_6:(5852, 1.502e+04] | bin_7:(1.502e+04, inf) | __MISSING__ |
| dpdmax_139P__max | numeric | 9 | 0.17238 | 0.27023 | 中 | keep | bin_0:(-inf, 0.5] | bin_1:(0.5, 3.5] | bin_2:(3.5, 10.5] | bin_3:(10.5, 14.5] | bin_4:(14.5, 22.5] | bin_5:(22.5, 29.5] | bin_6:(29.5, 48.5] | bin_7:(48.5, inf) | __MISSING__ |
| pmts_dpd_1073P_std__max | numeric | 9 | 0.183774 | 0.269176 | 中 | keep | bin_0:(-inf, 0.2157] | bin_1:(0.2157, 0.5911] | bin_2:(0.5911, 1.031] | bin_3:(1.031, 2.499] | bin_4:(2.499, 5.768] | bin_5:(5.768, 8.528] | bin_6:(8.528, 19.12] | bin_7:(19.12, inf) | __MISSING__ |
| pmts_dpd_303P_std__max | numeric | 9 | 0.341315 | 0.264389 | 中 | keep | bin_0:(-inf, 1.45] | bin_1:(1.45, 2.859] | bin_2:(2.859, 9.091] | bin_3:(9.091, 16.15] | bin_4:(16.15, 25.65] | bin_5:(25.65, 179.5] | bin_6:(179.5, 205.3] | bin_7:(205.3, inf) | __MISSING__ |
| pmts_overdue_1140A_std__mean | numeric | 9 | 0.182161 | 0.261742 | 中 | keep | bin_0:(-inf, 28.49] | bin_1:(28.49, 178.8] | bin_2:(178.8, 431.1] | bin_3:(431.1, 592.7] | bin_4:(592.7, 1078] | bin_5:(1078, 1590] | bin_6:(1590, 2240] | bin_7:(2240, inf) | __MISSING__ |
| pmts_overdue_1140A_mean__max | numeric | 9 | 0.173909 | 0.261496 | 中 | keep | bin_0:(-inf, 1.733] | bin_1:(1.733, 31.46] | bin_2:(31.46, 271.4] | bin_3:(271.4, 369.4] | bin_4:(369.4, 472.4] | bin_5:(472.4, 1411] | bin_6:(1411, 3367] | bin_7:(3367, inf) | __MISSING__ |
| pmts_dpd_1073P_mean__max | numeric | 9 | 0.175446 | 0.260974 | 中 | keep | bin_0:(-inf, 0.04881] | bin_1:(0.04881, 0.08893] | bin_2:(0.08893, 0.1752] | bin_3:(0.1752, 0.645] | bin_4:(0.645, 2.27] | bin_5:(2.27, 4.568] | bin_6:(4.568, 11.66] | bin_7:(11.66, inf) | __MISSING__ |
| maxdbddpdtollast12m_3658940P | numeric | 9 | 0.462123 | 0.260003 | 中 | keep | bin_0:(-inf, -2.5] | bin_1:(-2.5, 0.5] | bin_2:(0.5, 1.5] | bin_3:(1.5, 3.5] | bin_4:(3.5, 8.5] | bin_5:(8.5, 17.5] | bin_6:(17.5, 32.5] | bin_7:(32.5, inf) | __MISSING__ |
| pctinstlsallpaidlate4d_3546849L | numeric | 9 | 0.301198 | 0.259237 | 中 | keep | bin_0:(-inf, 0.001885] | bin_1:(0.001885, 0.03118] | bin_2:(0.03118, 0.07542] | bin_3:(0.07542, 0.1657] | bin_4:(0.1657, 0.283] | bin_5:(0.283, 0.3518] | bin_6:(0.3518, 0.5827] | bin_7:(0.5827, inf) | __MISSING__ |
| pmts_dpd_1073P_recent12_mean__mean | numeric | 9 | 0.175446 | 0.258539 | 中 | keep | bin_0:(-inf, 0.01472] | bin_1:(0.01472, 0.06206] | bin_2:(0.06206, 0.1399] | bin_3:(0.1399, 0.5678] | bin_4:(0.5678, 1.101] | bin_5:(1.101, 2.801] | bin_6:(2.801, 6.348] | bin_7:(6.348, inf) | __MISSING__ |
| dpdmax_757P__max | numeric | 9 | 0.336736 | 0.257855 | 中 | keep | bin_0:(-inf, 4.5] | bin_1:(4.5, 7.5] | bin_2:(7.5, 17.5] | bin_3:(17.5, 34.5] | bin_4:(34.5, 74.5] | bin_5:(74.5, 759.5] | bin_6:(759.5, 1202] | bin_7:(1202, inf) | __MISSING__ |
| pmts_dpd_303P_mean__max | numeric | 9 | 0.337875 | 0.256193 | 中 | keep | bin_0:(-inf, 0.02083] | bin_1:(0.02083, 0.4599] | bin_2:(0.4599, 1.297] | bin_3:(1.297, 2.698] | bin_4:(2.698, 5.659] | bin_5:(5.659, 7.262] | bin_6:(7.262, 21.44] | bin_7:(21.44, inf) | __MISSING__ |
| dpdmax_757P__std | numeric | 9 | 0.428849 | 0.254505 | 中 | keep | bin_0:(-inf, 0.2373] | bin_1:(0.2373, 0.5465] | bin_2:(0.5465, 5.762] | bin_3:(5.762, 10.79] | bin_4:(10.79, 12.61] | bin_5:(12.61, 27.82] | bin_6:(27.82, 409.3] | bin_7:(409.3, inf) | __MISSING__ |
| numberofoverdueinstlmax_1039L__mean | numeric | 9 | 0.172044 | 0.251043 | 中 | keep | bin_0:(-inf, 0.1339] | bin_1:(0.1339, 1.646] | bin_2:(1.646, 7.367] | bin_3:(7.367, 16.29] | bin_4:(16.29, 27.23] | bin_5:(27.23, 46.58] | bin_6:(46.58, 86.6] | bin_7:(86.6, inf) | __MISSING__ |
| numberofoverdueinstlmax_1151L__max | numeric | 9 | 0.335759 | 0.25098 | 中 | keep | bin_0:(-inf, 4.5] | bin_1:(4.5, 11.5] | bin_2:(11.5, 28.5] | bin_3:(28.5, 32.5] | bin_4:(32.5, 60.5] | bin_5:(60.5, 845.5] | bin_6:(845.5, 1300] | bin_7:(1300, inf) | __MISSING__ |
| numberofoverdueinstlmax_1151L__std | numeric | 9 | 0.42611 | 0.250959 | 中 | keep | bin_0:(-inf, 0.4986] | bin_1:(0.4986, 4.95] | bin_2:(4.95, 9.911] | bin_3:(9.911, 12.12] | bin_4:(12.12, 18.32] | bin_5:(18.32, 31.79] | bin_6:(31.79, 420.9] | bin_7:(420.9, inf) | __MISSING__ |
| avgdbddpdlast24m_3658932P | numeric | 9 | 0.401663 | 0.25083 | 中 | keep | bin_0:(-inf, -5.5] | bin_1:(-5.5, -4.5] | bin_2:(-4.5, -2.5] | bin_3:(-2.5, -0.5] | bin_4:(-0.5, 0.5] | bin_5:(0.5, 3.5] | bin_6:(3.5, 7.5] | bin_7:(7.5, inf) | __MISSING__ |

完整結果見 `data/base_final_woe_iv.csv`。
