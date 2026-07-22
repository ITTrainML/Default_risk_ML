"""從 base_final.parquet 保留最終選定欄位,依 WEEK_NUM(相異週數)切出 train/valid。

流程:
1. 以 KEEP_COLS(90 欄:case_id/target/WEEK_NUM + 87 個特徵)篩選 base_final.parquet。
2. 相異 WEEK_NUM 值由大到小排序,取最新 round(n_weeks*0.2) 週當驗證集,其餘當訓練集。
3. 輸出 data/train_model.parquet、data/valid_model.parquet。
"""

import duckdb
import polars as pl

SRC = "data/base_final.parquet"
TRAIN_OUT = "data/train_model.parquet"
VALID_OUT = "data/valid_model.parquet"

VALID_FRACTION = 0.2

KEEP_COLS = [
    "case_id",
    "target",
    "WEEK_NUM",
    "avgdpdtolclosure24_3658938P",
    "maxdbddpdtollast12m_3658940P",
    "pctinstlsallpaidlate1d_3546856L",
    "reject_rate",
    "pmts_dpd_1073P_trend__max",
    "pmts_dpd_303P_longest_good_streak__max",
    "pctinstlsallpaidearl3d_427L",
    "maxdpdtolerance_577P_mean",
    "rejectreasonclient_4145042M_non_placeholder_ratio",
    "lastrejectreason_759M",
    "pmts_overdue_1140A_overdue_rate__weighted_avg",
    "pmts_dpd_1073P_trend__mean",
    "pmts_dpd_303P_trend__mean",
    "cntpmts24_3658933L",
    "disbursedcredamount_1113A",
    "maxdpdlast12m_727P",
    "maxdpdlast24m_143P",
    "numrejects9m_859L",
    "price_1097A",
    "education_1103M",
    "pmtaverage_3A",
    "requesttype_4525192L",
    "tenure_years_max",
    "age_years_appl",
    "tenure_years_appl",
    "education_927M_appl",
    "incometype_1044T_appl",
    "numberofoutstandinstls_59L__min",
    "numberofoutstandinstls_59L__sum",
    "overdueamountmax_35A__mean",
    "overdueamountmax_35A__std",
    "pmts_dpd_303P_mean_positive__recomputed",
    "interestrate_311L",
    "maxdebt4_972A",
    "maxdpdtolerance_374P",
    "numinstlswithdpd10_728L",
    "numinstlswithdpd5_4187116L",
    "totalsettled_863A",
    "numberofoverdueinstlmax_1039L__min",
    "prolongationcount_599L__null_rate",
    "residualamount_856A__min",
    "totalamount_6A__sum",
    "totalamount_996A__mean",
    "pmts_dpd_303P_non_null_count__sum",
    "collater_valueofguarantee_1124L_null_count__max",
    "pmts_dpd_303P_recent_time_key__mean_fallback",
    "disbursementtype_67L",
    "maxdpdlast3m_392P",
    "numinstlswithoutdpd_562L",
    "numinstunpaidmaxest_4493212L",
    "days180_256L",
    "days30_165L",
    "days360_512L",
    "days90_310L",
    "amount_4527230A_sum_positive",
    "cancelreason_3545846M_mode",
    "education_1138M_mode",
    "familystate_447L_appl",
    "dpdmax_757P__null_rate",
    "monthlyinstlamount_674A__mean",
    "outstandingamount_362A__max",
    "overdueamountmax_155A__std",
    "maxannuity_4075009A",
    "empl_industry_691L_appl",
    "amount_4527230A_positive_count",
    "credtype_322L",
    "last_status",
    "riskassesment_940T",
    "b2_pmts_dpdvalue_108P_overdue_rate_recomputed",
    "b2_pmts_dpdvalue_108P_overdue_rate_contract_max",
    "b2_pmts_dpdvalue_108P_mean_weighted",
    "pmts_dpd_303P_n_unique__sum",
    "pmts_year_1139T_pmts_month_158T_duration__mean",
    "pmts_overdue_1140A_n_unique__sum",
    "pmts_dpd_1073P_recent_time_key__mean_fallback",
    "pmts_dpd_303P_recent_time_key__max_fallback",
    "opencred_647L",
    "subjectroles_name_838M_n_unique__sum",
    "subjectroles_name_838M_entropy__mean",
    "pmts_overdue_1140A_non_null_count__max",
    "pmts_dpd_303P_std__max",
    "registaddr_zipcode_184M_appl",
    "lastrejectdate_50D",
    "maxdpdinstldate_3546855D",
    "lastdelinqdate_224D",
    "pmts_year_507T_pmts_month_706T_min__min",
    "pmts_year_1139T_pmts_month_158T_min__min",
]


def main():
    assert len(KEEP_COLS) == len(set(KEEP_COLS)), "KEEP_COLS 有重複欄位"

    con = duckdb.connect()
    schema_cols = set(
        con.sql(f"DESCRIBE SELECT * FROM '{SRC}'").pl()["column_name"].to_list()
    )
    missing = [c for c in KEEP_COLS if c not in schema_cols]
    if missing:
        raise ValueError(f"以下欄位不存在於 {SRC}: {missing}")

    df = pl.scan_parquet(SRC).select(KEEP_COLS).collect()
    print(f"篩選後資料: {df.shape}")

    distinct_weeks = sorted(df["WEEK_NUM"].unique().to_list())
    n_weeks = len(distinct_weeks)
    n_valid_weeks = round(n_weeks * VALID_FRACTION)
    valid_weeks = set(distinct_weeks[-n_valid_weeks:])
    cutoff = min(valid_weeks)
    print(
        f"WEEK_NUM 相異值: {n_weeks} 週(範圍 {distinct_weeks[0]}~{distinct_weeks[-1]});"
        f"取最新 {n_valid_weeks} 週(WEEK_NUM >= {cutoff})當驗證集"
    )

    train = df.filter(pl.col("WEEK_NUM") < cutoff)
    valid = df.filter(pl.col("WEEK_NUM") >= cutoff)

    train.write_parquet(TRAIN_OUT)
    valid.write_parquet(VALID_OUT)

    total = df.height
    print(f"train: {train.shape}  ({train.height / total:.2%})  "
          f"WEEK_NUM [{train['WEEK_NUM'].min()}, {train['WEEK_NUM'].max()}]  "
          f"target 違約率={train['target'].mean():.4%}")
    print(f"valid: {valid.shape}  ({valid.height / total:.2%})  "
          f"WEEK_NUM [{valid['WEEK_NUM'].min()}, {valid['WEEK_NUM'].max()}]  "
          f"target 違約率={valid['target'].mean():.4%}")
    print(f"written: {TRAIN_OUT}, {VALID_OUT}")


if __name__ == "__main__":
    main()
