"""Train and compare delivery-time models on a chronological split.

Fixes over the exploratory notebook:
  - train/test split is chronological (order_date), not random, since the
    model predicts an ETA and must generalize to future orders.
  - the ordinal encoder leaves missing values as NaN instead of a -999
    sentinel, so the downstream KNN imputer doesn't treat -999 as a real
    distance.
  - reports a naive baseline (predict the training median) so model MAE/R2
    can be judged against a real floor, not in isolation.
  - Random Forest is lightly tuned with cross-validated random search
    instead of one fit on default hyperparameters.
  - models are compared with a champion/challenger tournament: each
    challenger must beat the current champion by a paired, statistically
    significant margin (A/B test) on the same held-out rows, not just by a
    numerically lower aggregate metric.
"""

import subprocess

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
from xgboost import XGBRegressor

from ab_test import compare_model_errors
from model_utils import build_ordinal_encoder, naive_median_predictions, time_based_split

set_config(transform_output="pandas")

DATA_PATH = "swiggy_cleaned.csv"
MODEL_PATH = "model.joblib"

COLUMNS_TO_DROP = [
    "rider_id",
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
    "order_time_hour",
    "order_day",
    # not available at prediction time (the order hasn't been picked up yet)
    "pickup_time_minutes",
]

NUM_COLS = ["age", "ratings", "distance"]
NOMINAL_CAT_COLS = [
    "weather", "type_of_order", "type_of_vehicle", "festival",
    "city_type", "city_name", "order_month", "order_day_of_week",
    "is_weekend", "order_time_of_day",
]
ORDINAL_CAT_COLS = ["traffic", "distance_type"]
TRAFFIC_ORDER = ["low", "medium", "high", "jam"]
DISTANCE_TYPE_ORDER = ["short", "medium", "long", "very_long"]
MODE_IMPUTE_COLS = ["multiple_deliveries", "festival", "city_type"]

# TabPFN's attention scales ~quadratically with the number of rows given as
# context, so unlike the tree models it isn't built for a training set this
# size. Cap training rows for it specifically (seeded, so reproducible);
# evaluation still runs on the full test set, so the tournament comparison
# against the other models stays valid.
TABPFN_MAX_TRAIN_SAMPLES = 10_000


def build_pipeline(model):
    missing_impute_cols = [c for c in NOMINAL_CAT_COLS if c not in MODE_IMPUTE_COLS]

    simple_imputer = ColumnTransformer(
        transformers=[
            ("mode_imputer", SimpleImputer(strategy="most_frequent"), MODE_IMPUTE_COLS),
            ("missing_imputer", SimpleImputer(strategy="constant", fill_value="missing"), missing_impute_cols),
        ],
        remainder="passthrough", n_jobs=-1,
        verbose_feature_names_out=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(), NUM_COLS),
            ("nominal_encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), NOMINAL_CAT_COLS),
            ("ordinal_encode", build_ordinal_encoder([TRAFFIC_ORDER, DISTANCE_TYPE_ORDER]), ORDINAL_CAT_COLS),
        ],
        remainder="passthrough", n_jobs=-1,
        verbose_feature_names_out=False,
    )

    processing_pipeline = Pipeline(steps=[
        ("simple_imputer", simple_imputer),
        ("preprocess", preprocessor),
        ("knn_imputer", KNNImputer(n_neighbors=5)),
    ])

    return Pipeline(steps=[("preprocessing", processing_pipeline), ("model", model)])


def detect_xgb_device() -> str:
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return "cuda"
    except Exception:
        return "cpu"


def evaluate(y_true, y_pred_transformed, power_transformer):
    y_pred = np.asarray(
        power_transformer.inverse_transform(np.asarray(y_pred_transformed).reshape(-1, 1))
    ).ravel()
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }, y_pred


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    df = df.drop(columns=COLUMNS_TO_DROP)

    train_df, test_df = time_based_split(df, date_col="order_date", test_size=0.2)
    train_df = train_df.drop(columns="order_date")
    test_df = test_df.drop(columns="order_date")

    X_train, y_train = train_df.drop(columns="time_taken"), train_df["time_taken"]
    X_test, y_test = test_df.drop(columns="time_taken"), test_df["time_taken"]
    print(f"train rows: {len(X_train)}, test rows: {len(X_test)} (chronological split)")

    pt = PowerTransformer()
    y_train_pt = np.asarray(pt.fit_transform(y_train.values.reshape(-1, 1))).ravel()

    # naive baseline: always predict the training median
    naive_preds = naive_median_predictions(y_train, n=len(y_test))
    naive_metrics = {
        "mae": mean_absolute_error(y_test, naive_preds),
        "rmse": root_mean_squared_error(y_test, naive_preds),
        "r2": r2_score(y_test, naive_preds),
    }
    print(f"\nNaive baseline (predict training median): {naive_metrics}")

    candidates = []  # (name, fitted_pipe, test_preds)

    # Linear Regression
    lr_pipe = build_pipeline(LinearRegression())
    lr_pipe.fit(X_train, y_train_pt)
    lr_metrics, lr_test_preds = evaluate(y_test, lr_pipe.predict(X_test), pt)
    lr_train_metrics, _ = evaluate(y_train, lr_pipe.predict(X_train), pt)
    print(f"\nLinear Regression train: {lr_train_metrics}")
    print(f"Linear Regression test:  {lr_metrics}")
    candidates.append(("linear_regression", lr_pipe, lr_test_preds))

    # Random Forest, lightly tuned with cross-validated random search
    rf_param_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.5, 1.0],
    }
    rf_pipe = build_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_distributions, n_iter=8, cv=3,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train_pt)
    rf_pipe = rf_search.best_estimator_
    print(f"\nRandom Forest best params: {rf_search.best_params_}")

    rf_metrics, rf_test_preds = evaluate(y_test, rf_pipe.predict(X_test), pt)
    rf_train_metrics, _ = evaluate(y_train, rf_pipe.predict(X_train), pt)
    print(f"Random Forest train: {rf_train_metrics}")
    print(f"Random Forest test:  {rf_metrics}")
    candidates.append(("random_forest", rf_pipe, rf_test_preds))

    # XGBoost - uses the GPU if this machine has one (e.g. a Colab GPU
    # runtime), falls back to CPU otherwise
    xgb_device = detect_xgb_device()
    print(f"\nXGBoost device: {xgb_device}")
    xgb_pipe = build_pipeline(XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device=xgb_device, random_state=42,
    ))
    xgb_pipe.fit(X_train, y_train_pt)
    xgb_metrics, xgb_test_preds = evaluate(y_test, xgb_pipe.predict(X_test), pt)
    xgb_train_metrics, _ = evaluate(y_train, xgb_pipe.predict(X_train), pt)
    print(f"XGBoost train: {xgb_train_metrics}")
    print(f"XGBoost test:  {xgb_metrics}")
    candidates.append(("xgboost", xgb_pipe, xgb_test_preds))

    # LightGBM - CPU only here; GPU LightGBM needs a special build not
    # guaranteed to be present on a stock machine/Colab image
    lgbm_pipe = build_pipeline(LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    ))
    lgbm_pipe.fit(X_train, y_train_pt)
    lgbm_metrics, lgbm_test_preds = evaluate(y_test, lgbm_pipe.predict(X_test), pt)
    lgbm_train_metrics, _ = evaluate(y_train, lgbm_pipe.predict(X_train), pt)
    print(f"\nLightGBM train: {lgbm_train_metrics}")
    print(f"LightGBM test:  {lgbm_metrics}")
    candidates.append(("lightgbm", lgbm_pipe, lgbm_test_preds))

    # TabPFN - a pretrained tabular foundation model (in-context learning,
    # no hyperparameter search). Subsample training rows since its
    # attention doesn't scale to a training set this size; test evaluation
    # still uses the full held-out set.
    if len(X_train) > TABPFN_MAX_TRAIN_SAMPLES:
        tabpfn_rng = np.random.RandomState(42)
        tabpfn_idx = tabpfn_rng.choice(len(X_train), size=TABPFN_MAX_TRAIN_SAMPLES, replace=False)
        X_train_tabpfn = X_train.iloc[tabpfn_idx]
        y_train_pt_tabpfn = y_train_pt[tabpfn_idx]
        y_train_tabpfn = y_train.iloc[tabpfn_idx]
    else:
        X_train_tabpfn = X_train
        y_train_pt_tabpfn = y_train_pt
        y_train_tabpfn = y_train
    print(f"\nTabPFN training rows: {len(X_train_tabpfn)} (of {len(X_train)} available)")

    tabpfn_pipe = build_pipeline(TabPFNRegressor(device=xgb_device))
    tabpfn_pipe.fit(X_train_tabpfn, y_train_pt_tabpfn)
    tabpfn_metrics, tabpfn_test_preds = evaluate(y_test, tabpfn_pipe.predict(X_test), pt)
    tabpfn_train_metrics, _ = evaluate(y_train_tabpfn, tabpfn_pipe.predict(X_train_tabpfn), pt)
    print(f"TabPFN train: {tabpfn_train_metrics}")
    print(f"TabPFN test:  {tabpfn_metrics}")
    candidates.append(("tabpfn", tabpfn_pipe, tabpfn_test_preds))

    # Champion/challenger tournament: each challenger must beat the current
    # champion by a statistically significant margin (paired Wilcoxon
    # signed-rank test on the same test rows) to take the title.
    champion_name, champion_pipe, champion_preds = candidates[0]
    champion_errors = np.abs(y_test.values - champion_preds)
    print("\nTournament:")
    for challenger_name, challenger_pipe, challenger_preds in candidates[1:]:
        challenger_errors = np.abs(y_test.values - challenger_preds)
        result = compare_model_errors(
            champion_errors, challenger_errors,
            model_a_name=champion_name, model_b_name=challenger_name,
        )
        print(f"  {champion_name} vs {challenger_name}: {result}")
        if result["better_model"] == challenger_name:
            champion_name, champion_pipe, champion_errors = challenger_name, challenger_pipe, challenger_errors

    print(f"\nFinal champion: {champion_name}")
    joblib.dump({"pipeline": champion_pipe, "power_transformer": pt}, MODEL_PATH)
    print(f"Saved winning model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
