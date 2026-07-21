import duckdb

con = duckdb.connect()

con.sql("""
    COPY (
        SELECT *
        FROM 'data/base.parquet' p1
        LEFT JOIN 'data/train_person_agg.parquet' b USING (case_id)
    ) TO 'data/base_final.parquet' (FORMAT PARQUET)
""")

print(con.sql("SELECT COUNT(*) AS n_rows FROM 'data/base_final.parquet'"))
print(con.sql("DESCRIBE SELECT * FROM 'data/base_final.parquet'"))
