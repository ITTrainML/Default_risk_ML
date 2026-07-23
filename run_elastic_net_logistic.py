# -*- coding: utf-8 -*-
import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


TRAIN_FILE = Path("train_base_final3.parquet")
TEST_FILE = Path("test_base_final3.parquet")
RECOMMENDATIONS_FILE = Path(
    "feature_shape_output/ebm_shape_diagnosis/feature_shape_recommendations.csv"
)
OUT_DIR = Path("elastic_net_logistic_output")

KEY_COL = "case_id"
TARGET_COL = "target"
WEEK_COL = "WEEK_NUM"
DROP_COLS = [
    "case_id",
    "target",
    "WEEK_NUM",
    "lastrejectdate_50D",
    "maxdpdinstldate_3546855D",
    "lastdelinqdate_224D",
]

WOE_ACTIONS = {"bin_or_woe", "monotonic_bin_or_woe", "keep_nominal_or_woe"}
POLY_ACTIONS = {"polynomial_or_spline"}
SPLINE_ACTIONS = {"spline_or_tree_raw"}
RAW_ACTIONS = {"keep_raw", "keep_low_priority", "review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Elastic Net Logistic model and evaluate weekly Gini stability."
    )
    parser.add_argument("--train", default=str(TRAIN_FILE))
    parser.add_argument("--test", default=str(TEST_FILE))
    parser.add_argument("--recommendations", default=str(RECOMMENDATIONS_FILE))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--l1-c", type=float, default=1.0)
    parser.add_argument("--elastic-c", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--woe-bins", type=int, default=10)
    parser.add_argument("--onehot-max-unique", type=int, default=50)
    parser.add_argument("--spline-knots", type=int, default=5)
    parser.add_argument("--coef-epsilon", type=float, default=1e-8)
    parser.add_argument("--random-state", type=int, default=20260720)
    parser.add_argument("--sample-rows", type=int, default=None)
    return parser.parse_args()


def read_frame(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=20260720).sort_index()
    return df


def load_recommendations(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rec = pd.read_csv(path)
    if "feature" not in rec.columns or "recommended_action" not in rec.columns:
        return {}
    return dict(zip(rec["feature"].astype(str), rec["recommended_action"].astype(str)))


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    valid = y_true.notna() & pd.notna(y_score)
    y = y_true[valid]
    score = y_score[valid.to_numpy()]
    if y.nunique(dropna=True) < 2:
        return None
    return float(roc_auc_score(y.astype(int), score))


def gini_from_auc(auc: float | None) -> float | None:
    return None if auc is None else float(2.0 * auc - 1.0)


def metric_table(df: pd.DataFrame, pred_col: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for week, week_df in df.groupby(WEEK_COL, dropna=False):
        auc = safe_auc(week_df[TARGET_COL], week_df[pred_col].to_numpy())
        rows.append(
            {
                WEEK_COL: week,
                "row_count": int(len(week_df)),
                "target_count": int(week_df[TARGET_COL].notna().sum()),
                "target_positive": int((week_df[TARGET_COL] == 1).sum()),
                "target_negative": int((week_df[TARGET_COL] == 0).sum()),
                "auc": auc,
                "gini": gini_from_auc(auc),
                "warning": "single_target_class" if auc is None else "",
            }
        )
    weekly = pd.DataFrame(rows).sort_values(WEEK_COL)
    valid = weekly.dropna(subset=["gini", WEEK_COL]).copy()
    overall_auc = safe_auc(df[TARGET_COL], df[pred_col].to_numpy())
    summary = {
        "auc": overall_auc,
        "gini": gini_from_auc(overall_auc),
        "weekly_gini_mean": None,
        "weekly_gini_slope": None,
        "falling_rate": None,
        "residual_std": None,
        "stability_metric": None,
        "valid_week_count": int(len(valid)),
    }
    if len(valid) >= 2:
        x = pd.to_numeric(valid[WEEK_COL], errors="coerce").to_numpy(dtype=float)
        y = valid["gini"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(y) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            residuals = y - (slope * x + intercept)
            falling_rate = min(0.0, float(slope))
            residual_std = float(np.std(residuals, ddof=0))
            summary.update(
                {
                    "weekly_gini_mean": float(np.mean(y)),
                    "weekly_gini_slope": float(slope),
                    "falling_rate": falling_rate,
                    "residual_std": residual_std,
                    "stability_metric": float(
                        np.mean(y) + 88.0 * falling_rate - 0.5 * residual_std
                    ),
                }
            )
    elif len(valid) == 1:
        summary["weekly_gini_mean"] = float(valid["gini"].iloc[0])
    return weekly, summary


@dataclass
class WoeSpec:
    feature: str
    is_numeric_feature: bool
    bins: np.ndarray | None
    mapping: dict
    default: float


class FeatureBuilder:
    def __init__(
        self,
        recommendations: dict[str, str],
        woe_bins: int,
        onehot_max_unique: int,
        spline_knots: int,
    ) -> None:
        self.recommendations = recommendations
        self.woe_bins = woe_bins
        self.onehot_max_unique = onehot_max_unique
        self.spline_knots = spline_knots
        self.raw_numeric: list[str] = []
        self.poly_numeric: list[str] = []
        self.spline_numeric: list[str] = []
        self.woe_features: list[str] = []
        self.onehot_features: list[str] = []
        self.medians: dict[str, float] = {}
        self.scaler: StandardScaler | None = None
        self.spline_transformers: dict[str, SplineTransformer] = {}
        self.spline_fallback_features: set[str] = set()
        self.woe_specs: dict[str, WoeSpec] = {}
        self.onehot_encoder: OneHotEncoder | None = None
        self.scaled_feature_names: list[str] = []
        self.other_feature_names: list[str] = []
        self.warnings: list[str] = []

    def action_for(self, feature: str) -> str:
        return self.recommendations.get(feature, "keep_raw")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> sparse.csr_matrix:
        for col in x.columns:
            numeric = is_numeric(x[col])
            action = self.action_for(col)
            if numeric and action in POLY_ACTIONS:
                self.poly_numeric.append(col)
            elif numeric and action in SPLINE_ACTIONS:
                self.spline_numeric.append(col)
            elif action in WOE_ACTIONS:
                self.woe_features.append(col)
            elif numeric or action in RAW_ACTIONS:
                if numeric:
                    self.raw_numeric.append(col)
                else:
                    self._add_categorical(col, x[col])
            else:
                self._add_categorical(col, x[col])

        for col in self.raw_numeric + self.poly_numeric + self.spline_numeric:
            values = clean_numeric(x[col])
            median = values.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        scaled_blocks = self._fit_scaled_blocks(x)
        other_blocks = self._fit_other_blocks(x, y)
        return self._combine(scaled_blocks, other_blocks)

    def transform(self, x: pd.DataFrame) -> sparse.csr_matrix:
        scaled_blocks = self._transform_scaled_blocks(x)
        other_blocks = self._transform_other_blocks(x)
        return self._combine(scaled_blocks, other_blocks)

    def feature_names(self) -> list[str]:
        names = list(self.scaled_feature_names) + list(self.other_feature_names)
        if not names:
            raise RuntimeError("no usable model features were generated")
        return names

    def _add_categorical(self, col: str, series: pd.Series) -> None:
        n_unique = series.nunique(dropna=True)
        if n_unique <= self.onehot_max_unique:
            self.onehot_features.append(col)
        else:
            self.woe_features.append(col)

    def _fit_scaled_blocks(self, x: pd.DataFrame) -> list[np.ndarray]:
        raw_blocks, names = self._make_unscaled_numeric_blocks(x, fit=True)
        if not raw_blocks:
            self.scaler = None
            self.scaled_feature_names = []
            return []
        raw_matrix = np.column_stack(raw_blocks)
        self.scaler = StandardScaler()
        self.scaled_feature_names = names
        return [self.scaler.fit_transform(raw_matrix)]

    def _transform_scaled_blocks(self, x: pd.DataFrame) -> list[np.ndarray]:
        raw_blocks, _ = self._make_unscaled_numeric_blocks(x, fit=False)
        if not raw_blocks or self.scaler is None:
            return []
        raw_matrix = np.column_stack(raw_blocks)
        return [self.scaler.transform(raw_matrix)]

    def _make_unscaled_numeric_blocks(
        self, x: pd.DataFrame, fit: bool
    ) -> tuple[list[np.ndarray], list[str]]:
        blocks = []
        names = []
        for col in self.raw_numeric:
            filled, missing = self._filled_numeric(x[col], col)
            blocks.extend([filled, missing])
            names.extend([col, f"{col}__missing"])
        for col in self.poly_numeric:
            filled, missing = self._filled_numeric(x[col], col)
            blocks.extend([filled, np.square(filled), missing])
            names.extend([col, f"{col}__poly2", f"{col}__missing"])
        for col in self.spline_numeric:
            if col in self.spline_fallback_features:
                continue
            filled, missing = self._filled_numeric(x[col], col)
            if fit:
                transformer = SplineTransformer(
                    n_knots=self.spline_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="constant",
                )
                try:
                    spline_values = transformer.fit_transform(filled.reshape(-1, 1))
                    self.spline_transformers[col] = transformer
                except Exception as exc:
                    self.warnings.append(f"{col}: spline fallback to raw ({exc})")
                    self.spline_fallback_features.add(col)
                    blocks.extend([filled, missing])
                    names.extend([col, f"{col}__missing"])
                    continue
            else:
                transformer = self.spline_transformers.get(col)
                if transformer is None:
                    blocks.extend([filled, missing])
                    names.extend([col, f"{col}__missing"])
                    continue
                spline_values = transformer.transform(filled.reshape(-1, 1))
            for idx in range(spline_values.shape[1]):
                blocks.append(spline_values[:, idx])
                names.append(f"{col}__spline_{idx + 1}")
            blocks.append(missing)
            names.append(f"{col}__missing")
        return blocks, names

    def _filled_numeric(self, series: pd.Series, col: str) -> tuple[np.ndarray, np.ndarray]:
        values = clean_numeric(series)
        missing = values.isna().astype(float).to_numpy()
        filled = values.fillna(self.medians[col]).to_numpy(dtype=float).copy()
        filled[~np.isfinite(filled)] = self.medians[col]
        return filled, missing

    def _fit_other_blocks(self, x: pd.DataFrame, y: pd.Series) -> list[sparse.csr_matrix]:
        blocks = []
        self.other_feature_names = []
        if self.woe_features:
            woe_matrix = self._fit_woe(x, y)
            blocks.append(sparse.csr_matrix(woe_matrix))
            self.other_feature_names.extend([f"{c}__woe" for c in self.woe_features])
        if self.onehot_features:
            cat = x[self.onehot_features].astype("string").fillna("__MISSING__")
            self.onehot_encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, min_frequency=None
            )
            encoded = self.onehot_encoder.fit_transform(cat)
            blocks.append(encoded)
            names = self.onehot_encoder.get_feature_names_out(self.onehot_features)
            self.other_feature_names.extend([str(name) for name in names])
        return blocks

    def _transform_other_blocks(self, x: pd.DataFrame) -> list[sparse.csr_matrix]:
        blocks = []
        if self.woe_features:
            blocks.append(sparse.csr_matrix(self._transform_woe(x)))
        if self.onehot_features and self.onehot_encoder is not None:
            cat = x[self.onehot_features].astype("string").fillna("__MISSING__")
            blocks.append(self.onehot_encoder.transform(cat))
        return blocks

    def _fit_woe(self, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
        matrices = []
        y_int = y.astype(int)
        total_event = float((y_int == 1).sum())
        total_non_event = float((y_int == 0).sum())
        for col in self.woe_features:
            numeric = is_numeric(x[col])
            labels, bins = self._woe_labels(x[col], numeric, fit=True)
            mapping = {}
            grouped = pd.DataFrame({"label": labels, "target": y_int}).groupby("label")
            for label, group in grouped:
                event = float((group["target"] == 1).sum())
                non_event = float((group["target"] == 0).sum())
                event_dist = (event + 0.5) / (total_event + 0.5 * len(grouped))
                non_event_dist = (non_event + 0.5) / (
                    total_non_event + 0.5 * len(grouped)
                )
                mapping[str(label)] = float(math.log(non_event_dist / event_dist))
            default = float(np.mean(list(mapping.values()))) if mapping else 0.0
            self.woe_specs[col] = WoeSpec(col, numeric, bins, mapping, default)
            matrices.append(self._labels_to_woe(labels, mapping, default))
        return np.column_stack(matrices) if matrices else np.empty((len(x), 0))

    def _transform_woe(self, x: pd.DataFrame) -> np.ndarray:
        matrices = []
        for col in self.woe_features:
            spec = self.woe_specs[col]
            labels, _ = self._woe_labels(x[col], spec.is_numeric_feature, fit=False, bins=spec.bins)
            matrices.append(self._labels_to_woe(labels, spec.mapping, spec.default))
        return np.column_stack(matrices) if matrices else np.empty((len(x), 0))

    def _woe_labels(
        self,
        series: pd.Series,
        numeric: bool,
        fit: bool,
        bins: np.ndarray | None = None,
    ) -> tuple[pd.Series, np.ndarray | None]:
        if numeric:
            values = clean_numeric(series)
            if fit:
                quantiles = np.linspace(0, 1, self.woe_bins + 1)
                raw_bins = np.nanquantile(values.to_numpy(dtype=float), quantiles)
                bins = np.unique(raw_bins[np.isfinite(raw_bins)])
                if bins is None or len(bins) < 3:
                    labels = values.fillna("__MISSING__").astype("string")
                    return labels, None
            if bins is not None and len(bins) >= 3:
                cut = pd.cut(values, bins=bins, include_lowest=True, duplicates="drop")
                labels = cut.astype("string").fillna("__MISSING__")
                return labels, bins
            labels = values.fillna("__MISSING__").astype("string")
            return labels, None
        return series.astype("string").fillna("__MISSING__"), None

    @staticmethod
    def _labels_to_woe(labels: pd.Series, mapping: dict, default: float) -> np.ndarray:
        return labels.astype(str).map(mapping).fillna(default).to_numpy(dtype=float)

    @staticmethod
    def _combine(
        scaled_blocks: list[np.ndarray], other_blocks: list[sparse.csr_matrix]
    ) -> sparse.csr_matrix:
        blocks = []
        for block in scaled_blocks:
            blocks.append(sparse.csr_matrix(block))
        blocks.extend(other_blocks)
        if not blocks:
            raise RuntimeError("no feature blocks generated")
        return sparse.hstack(blocks, format="csr")


def validate_inputs(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    warnings_out = []
    for required in [KEY_COL, TARGET_COL, WEEK_COL]:
        if required not in train.columns:
            raise RuntimeError(f"train is missing required column: {required}")
    for required in [KEY_COL, WEEK_COL]:
        if required not in test.columns:
            raise RuntimeError(f"test is missing required column: {required}")
    for col in DROP_COLS:
        if col not in train.columns:
            warnings_out.append(f"train missing drop column: {col}")
        if col not in test.columns and col != TARGET_COL:
            warnings_out.append(f"test missing drop column: {col}")
    return warnings_out


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in DROP_COLS if col in df.columns])


def assert_clean_matrix(name: str, matrix: sparse.csr_matrix) -> None:
    data = matrix.data
    if np.isnan(data).any() or np.isinf(data).any():
        raise RuntimeError(f"{name} contains NaN or Inf after preprocessing")


def evaluate_and_write(
    name: str,
    df: pd.DataFrame,
    pred: np.ndarray,
    output_dir: Path,
) -> dict | None:
    pred_df = pd.DataFrame({KEY_COL: df[KEY_COL], WEEK_COL: df[WEEK_COL], "prediction": pred})
    if TARGET_COL in df.columns:
        pred_df[TARGET_COL] = df[TARGET_COL]
    pred_df.to_parquet(output_dir / f"{name}_predictions.parquet", index=False)
    if TARGET_COL not in df.columns:
        return None
    weekly, summary = metric_table(pred_df, "prediction")
    weekly.to_csv(output_dir / f"{name}_weekly_metrics.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    train = read_frame(Path(args.train), args.sample_rows)
    test = read_frame(Path(args.test), args.sample_rows)
    run_warnings = validate_inputs(train, test)

    y_train = train[TARGET_COL].astype(int)
    x_train = model_input(train)
    x_test = model_input(test)
    x_test = x_test.reindex(columns=x_train.columns)

    recommendations = load_recommendations(Path(args.recommendations))
    builder = FeatureBuilder(
        recommendations=recommendations,
        woe_bins=args.woe_bins,
        onehot_max_unique=args.onehot_max_unique,
        spline_knots=args.spline_knots,
    )
    x_train_matrix = builder.fit(x_train, y_train)
    x_test_matrix = builder.transform(x_test)
    assert_clean_matrix("x_train", x_train_matrix)
    assert_clean_matrix("x_test", x_test_matrix)

    l1 = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=1.0,
        C=args.l1_c,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    l1.fit(x_train_matrix, y_train)
    feature_names = np.array(builder.feature_names(), dtype=object)
    l1_coef = l1.coef_[0]
    selected_mask = np.abs(l1_coef) > args.coef_epsilon
    if not selected_mask.any():
        run_warnings.append("L1 selected no features; fallback to all preprocessed features")
        selected_mask = np.ones_like(l1_coef, dtype=bool)

    selected_names = feature_names[selected_mask]
    dropped_names = feature_names[~selected_mask]
    pd.DataFrame({"feature": selected_names}).to_csv(
        output_dir / "selected_features.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"feature": dropped_names}).to_csv(
        output_dir / "dropped_by_l1_features.csv", index=False, encoding="utf-8-sig"
    )

    final_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=args.l1_ratio,
        C=args.elastic_c,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    final_model.fit(x_train_matrix[:, selected_mask], y_train)

    train_pred = final_model.predict_proba(x_train_matrix[:, selected_mask])[:, 1]
    test_pred = final_model.predict_proba(x_test_matrix[:, selected_mask])[:, 1]

    train_summary = evaluate_and_write("train", train, train_pred, output_dir)
    test_summary = evaluate_and_write("test", test, test_pred, output_dir)

    overall_rows = []
    if train_summary is not None:
        overall_rows.append({"dataset": "train", **train_summary})
    if test_summary is not None:
        overall_rows.append({"dataset": "test", **test_summary})
    pd.DataFrame(overall_rows).to_csv(
        output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig"
    )

    coef = final_model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": selected_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_file": str(Path(args.train)),
        "test_file": str(Path(args.test)),
        "recommendations_file": str(Path(args.recommendations)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "input_feature_count": int(x_train.shape[1]),
        "preprocessed_feature_count": int(len(feature_names)),
        "selected_feature_count": int(len(selected_names)),
        "dropped_by_l1_feature_count": int(len(dropped_names)),
        "raw_numeric_count": len(builder.raw_numeric),
        "polynomial_numeric_count": len(builder.poly_numeric),
        "spline_numeric_count": len(builder.spline_numeric),
        "woe_feature_count": len(builder.woe_features),
        "onehot_feature_count": len(builder.onehot_features),
        "train_metrics": train_summary,
        "test_metrics": test_summary,
        "warnings": run_warnings + builder.warnings,
        "parameters": vars(args),
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
