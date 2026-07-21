import duckdb

con = duckdb.connect()

con.sql("""
    COPY (
        SELECT p1.*, p2.* EXCLUDE (case_id, num_group1), b.date_decision
        FROM 'data/train_person_1.parquet' p1
        LEFT JOIN 'data/train_person_2_aggregated.parquet' p2 USING (case_id, num_group1)
        LEFT JOIN 'data/train_base.parquet' b USING (case_id)
    ) TO 'data/train_person_2_join_1.parquet' (FORMAT PARQUET)
""")

print(con.sql("SELECT COUNT(*) AS n_rows FROM 'data/train_person_2_join_1.parquet'"))
print(con.sql("DESCRIBE SELECT * FROM 'data/train_person_2_join_1.parquet'"))
