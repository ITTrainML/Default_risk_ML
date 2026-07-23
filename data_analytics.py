import duckdb

con = duckdb.connect()

# 確認沒有重複case_id



con.sql("""
    CREATE OR REPLACE VIEW train_static_0_0_DiffA AS
    SELECT case_id,count(case_id) as c_case_id
    FROM 'data/train/train_static_0_0.parquet'
    group by case_id
    having c_case_id >1
""")

con.sql("""
    CREATE OR REPLACE VIEW train_base_DiffA AS
    SELECT case_id,count(case_id) as c_case_id
    FROM 'data/train/train_base.parquet'
    group by case_id
    having c_case_id >1
""")

#確認筆數,max
con.sql("""
    CREATE OR REPLACE VIEW train_static_0_1 AS
    SELECT *
    FROM 'data/train/train_static_0_1.parquet'
""")

con.sql("""
    CREATE OR REPLACE VIEW train_static_0_0 AS
    SELECT *
    FROM 'data/train/train_static_0_0.parquet'
""")


# con.sql("""
#     CREATE OR REPLACE VIEW train_applprev1 AS
#     SELECT *
#     FROM 'data/train_applprev1_0.parquet'
#     Union
#     SELECT *
#     FROM 'data/train_applprev1_0.parquet'
# """)

con.sql("""
    CREATE OR REPLACE VIEW df_train AS
    SELECT *
    FROM 'data/df_train.parquet'
""")

con.sql("""
    CREATE OR REPLACE VIEW valid_base_final AS
    SELECT *
    FROM 'data/valid_base_final.parquet'
""")

con.sql("CALL start_ui();")
print(con.sql("SELECT loaded,installed FROM duckdb_extensions() WHERE extension_name = 'ui';"))

#叫出duckui，請在下列程式中設定中斷點
print("Hellow")

#test
