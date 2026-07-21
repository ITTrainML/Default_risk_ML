import math
import duckdb
import polars as pl

con = duckdb.connect()

con.sql("""
        CREATE VIEW train_base AS
        select *
        from 'data/base.parquet'
""")

con.sql("""
    CREATE VIEW train_applprev_join AS
    SELECT *
    FROM 'data/train_applprev_join.parquet'
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