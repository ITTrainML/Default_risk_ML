import polars as pl
import lightgbm as lgb
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    log_loss,
    brier_score_loss
)

train = pl.scan_parquet("train_base_final3.parquet")
test = pl.scan_parquet("test_base_final3.parquet")

x_train = train.drop(["case_id","target","WEEK_NUM","lastrejectdate_50D","maxdpdinstldate_3546855D","lastdelinqdate_224D"])
x_test = test.drop(["case_id","target","WEEK_NUM","lastrejectdate_50D","maxdpdinstldate_3546855D","lastdelinqdate_224D"])
y_train = train.select("target")
y_test = test.select("target")

week_train_pd = (
    train
    .select("WEEK_NUM")
    .collect()
    .to_numpy()
    .ravel()
)

week_test_pd = (
    test
    .select("WEEK_NUM")
    .collect()
    .to_numpy()
    .ravel()
)


x_train_pd = x_train.collect().to_pandas()
y_train_pd = y_train.collect().to_numpy().ravel()
x_test_pd = x_test.collect().to_pandas()
y_test_pd = y_test.collect().to_numpy().ravel()


categorical_cols = list(
    x_train_pd.select_dtypes(include=["object"]).columns
)

for col in categorical_cols:
    x_train_pd[col] = x_train_pd[col].astype("category")
    x_test_pd[col] = x_test_pd[col].astype("category")

    model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=31,

    random_state=42,

    n_jobs=-1
)

model.fit(
    x_train_pd,
    y_train_pd
)

## 這個就是輸出機率

train_probability = model.predict_proba(x_train_pd)[:,1]
test_probability = model.predict_proba(x_test_pd)[:,1]