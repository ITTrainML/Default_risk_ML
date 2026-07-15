import math
import duckdb
import polars as pl

con = duckdb.connect()

con.sql("""
    COPY(
        select *
        from(
        select * from 'data/train_applprev_1_0.parquet'
        union 
        select * from 'data/train_applprev_1_1.parquet') as a1
        left join
        'data/train_applprev_2_agg.parquet' as a2
        on a1.case_id = a2.case_id and a1.num_group1 = a2.num_group1
        )to 'data/train_applprev_join.parquet' (FORMAT PARQUET)

""")

con.sql("""
    CREATE VIEW train_applprev_join AS
    SELECT *
    FROM 'data/train_applprev_join.parquet'
""")

con.sql("""
    CREATE VIEW my_table AS
    SELECT *
    FROM 'data/train_applprev_2_agg.parquet'
""")

con.sql("CALL start_ui();")
print(con.sql("SELECT loaded,installed FROM duckdb_extensions() WHERE extension_name = 'ui';"))

#叫出duckui，請在下列程式中設定中斷點
print("Hellow")