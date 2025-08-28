from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ModelResult:
    name: str
    rmse: float
    mape: float
    r2: float
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(mape)


@st.cache_resource(show_spinner=False)
def train_compare(
    df: pd.DataFrame, target: str, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Train and compare multiple models using TimeSeriesSplit.

    Returns metrics_df, fitted_models, predictions_df (indexed by date if present).
    """
    if df is None or df.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    if not feature_cols:
        return pd.DataFrame(), {}, pd.DataFrame()
    data = df.copy()
    # Drop rows with NA in target or features
    cols = [target] + feature_cols
    data = data.dropna(subset=[c for c in cols if c in data.columns])
    if data.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    X = data[feature_cols].to_numpy()
    y = data[target].to_numpy()
    n_samples = len(data)
    n_features = X.shape[1] if X is not None else 0
    if n_samples == 0 or n_features == 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    # Pipelines
    numeric_features = list(range(X.shape[1]))
    pre = ColumnTransformer(
        [("num", StandardScaler(with_mean=False), numeric_features)], remainder="drop"
    )

    candidates = {
        "Linear": Pipeline([("scale", pre), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scale", pre), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline(
            [("scale", pre), ("model", Lasso(alpha=0.001, max_iter=5000))]
        ),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results: List[ModelResult] = []
    if n_samples <= 2:
        # Fit on full data; metrics as NaN
        for name, est in candidates.items():
            try:
                est.fit(X, y)
                pred_full = est.predict(X)
                results.append(
                    ModelResult(
                        name=name,
                        rmse=np.nan,
                        mape=np.nan,
                        r2=np.nan,
                        model=est,
                        y_true=y.copy(),
                        y_pred=np.array(pred_full),
                    )
                )
            except Exception:
                results.append(
                    ModelResult(
                        name=name,
                        rmse=np.nan,
                        mape=np.nan,
                        r2=np.nan,
                        model=est,
                        y_true=y.copy(),
                        y_pred=np.array([]),
                    )
                )
    else:
        # TimeSeriesSplit with valid number of splits
        n_splits = min(5, max(2, n_samples - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for name, est in candidates.items():
            y_true_all: List[float] = []
            y_pred_all: List[float] = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                est.fit(X_train, y_train)
                pred = est.predict(X_test)
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(pred.tolist())
            rmse = (
                float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
                if y_true_all
                else np.nan
            )
            r2 = float(r2_score(y_true_all, y_pred_all)) if y_true_all else np.nan
            mape = (
                _mape(np.array(y_true_all), np.array(y_pred_all))
                if y_true_all
                else np.nan
            )
            results.append(
                ModelResult(
                    name=name,
                    rmse=rmse,
                    mape=mape,
                    r2=r2,
                    model=est,
                    y_true=np.array(y_true_all),
                    y_pred=np.array(y_pred_all),
                )
            )

    # Pick best by RMSE
    results_sorted = sorted(
        results, key=lambda r: (np.inf if np.isnan(r.rmse) else r.rmse)
    )
    metrics_df = pd.DataFrame(
        [
            {"Model": r.name, "RMSE": r.rmse, "MAPE": r.mape, "R2": r.r2}
            for r in results_sorted
        ]
    )

    # Retrain all on full data for predictions
    fitted: Dict[str, object] = {}
    preds_df = pd.DataFrame(index=data.index)
    if "date" in data.columns:
        preds_df["date"] = data["date"].values
    for r in results_sorted:
        est = r.model
        est.fit(X, y)
        fitted[r.name] = est
        preds_df[r.name] = est.predict(X)
    preds_df[target] = y

    return metrics_df, fitted, preds_df
