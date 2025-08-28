"""Production Dashboard (Streamlit)
Monte Carlo simulation removed.
Run with: streamlit run app.py
"""

import io
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features import add_features
from modeling import train_compare

st.set_page_config(page_title="Production Dashboard", layout="wide")


# ----------------- Helpers -----------------
@st.cache_data(show_spinner=False)
def parse_csv(file_bytes) -> pd.DataFrame:
    return pd.read_csv(file_bytes, thousands=",", keep_default_na=True)


def coerce_dates(df: pd.DataFrame):
    cand = [c for c in df.columns if "date" in c.lower()]
    col = cand[0] if cand else df.columns[0]
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df, col


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(
                out[c].astype(str).str.replace(",", ""), errors="ignore"
            )
    return out


@st.cache_data(show_spinner=False)
def detect_outliers(
    df: pd.DataFrame, cols: List[str], method="iqr", z_thresh=3.0, iqr_mult=1.5
):
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if method == "zscore":
            m, sd = vals.mean(), vals.std(ddof=0)
            if sd == 0 or np.isnan(sd):
                continue
            flags[c] = np.abs((vals - m) / sd) > z_thresh
        else:  # IQR
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            flags[c] = (vals < lo) | (vals > hi)
    return flags


def resample_df(df: pd.DataFrame, date_col: str, rule: str) -> pd.DataFrame:
    if not rule:
        return df
    return df.set_index(date_col).resample(rule).mean(numeric_only=True).reset_index()


def build_time_feature(df: pd.DataFrame, date_col: str):
    base = df[date_col].min()
    return (
        ((df[date_col] - base).dt.total_seconds() / (24 * 3600))
        .to_numpy()
        .reshape(-1, 1)
    )


def fit_models(x, y, test_size=0.2, poly_degree=3, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    models = {
        "Linear": LinearRegression(),
        "Poly": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ("lin", LinearRegression()),
            ]
        ),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    metrics_rows = []
    preds_df = pd.DataFrame(
        {"split": ["train"] * len(x_train) + ["test"] * len(x_test)}
    )
    for name, mdl in models.items():
        mdl.fit(x_train, y_train)
        pred_train = mdl.predict(x_train)
        pred_test = mdl.predict(x_test)
        preds_df[name] = list(pred_train) + list(pred_test)
        if len(y_test):
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            mae = mean_absolute_error(y_test, pred_test)
            r2 = r2_score(y_test, pred_test)
        else:
            rmse = mae = r2 = np.nan
        metrics_rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(metrics_rows).sort_values("RMSE"), preds_df, models


def make_line_plot(
    df: pd.DataFrame,
    date_col: str,
    cols: List[str],
    show_outliers: Optional[pd.DataFrame] = None,
    ma_cols: Optional[List[str]] = None,
    ma_window: int = 1,
):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[c], mode="lines+markers", name=c))
        if ma_window > 1 and ma_cols and c in ma_cols:
            ma = (
                pd.to_numeric(df[c], errors="coerce")
                .rolling(ma_window, min_periods=1)
                .mean()
            )
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=ma,
                    mode="lines",
                    name=f"{c} MA({ma_window})",
                    line=dict(dash="dash"),
                )
            )
        if show_outliers is not None and c in show_outliers.columns:
            mask = show_outliers[c]
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, date_col],
                        y=df.loc[mask, c],
                        mode="markers",
                        name=f"{c} outliers",
                        marker=dict(color="red", symbol="triangle-up", size=10),
                    )
                )
    fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------- UI -----------------
st.title("Production Dashboard")
st.caption("Trend visualization & simple modeling")
st.sidebar.header("Controls")

_template = pd.DataFrame(
    {
        "Date": ["01/01/2019", "02/01/2019"],
        "OIL": [18659, 19771],
        "Water": [25311, 25825],
        "Gas": [49, 51],
    }
)
buf = io.StringIO()
_template.to_csv(buf, index=False)
st.sidebar.download_button(
    "⬇️ Template CSV", buf.getvalue().encode("utf-8"), file_name="template.csv"
)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Using template data until you upload.")
    df = _template.copy()
else:
    df = parse_csv(uploaded)

df, detected = coerce_dates(df)
df = numericize(df)

date_col = st.sidebar.selectbox(
    "Date column",
    options=df.columns.tolist(),
    index=df.columns.get_loc(detected),
    key="date_column_select",
)
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df = df.dropna(subset=[date_col])

numeric_cols = [
    c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])
]
y_cols = st.sidebar.multiselect("Series", options=numeric_cols, default=numeric_cols)

with st.sidebar.expander("Clean", expanded=False):
    strategy = st.selectbox(
        "Fill missing",
        ["None", "Forward fill", "Backward fill", "Interpolate"],
        index=0,
        key="fill_missing_strategy",
    )
    if strategy != "None":
        nums = [
            c
            for c in df.columns
            if c != date_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if strategy == "Forward fill":
            df[nums] = df[nums].ffill()
        elif strategy == "Backward fill":
            df[nums] = df[nums].bfill()
        else:
            df[nums] = df[nums].interpolate(limit_direction="both")

with st.sidebar.expander("Smoothing", expanded=False):
    ma_window = st.slider("MA window", 1, 30, 1, 1)
    apply_ma_cols = st.multiselect(
        "Apply MA to", options=y_cols, default=y_cols if ma_window > 1 else []
    )

if len(df):
    dmin, dmax = df[date_col].min(), df[date_col].max()
else:
    dmin = dmax = pd.Timestamp.today()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(dmin.date(), dmax.date()),
    min_value=dmin.date(),
    max_value=dmax.date(),
)
df = df[
    (df[date_col] >= pd.to_datetime(start_date))
    & (df[date_col] <= pd.to_datetime(end_date))
].sort_values(date_col)

freq_map = {"None": "", "Daily": "D", "Weekly": "W", "Monthly": "M"}
freq_choice = st.sidebar.selectbox(
    "Resample", list(freq_map.keys()), index=0, key="resample_freq"
)
df_res = resample_df(df, date_col, freq_map[freq_choice])

st.sidebar.subheader("Outliers")
method = st.sidebar.selectbox(
    "Method", ["zscore", "iqr"], index=1, key="outlier_method"
)
z_t = (
    st.sidebar.slider("Z threshold", 2.0, 5.0, 3.0, 0.1) if method == "zscore" else None
)
iqr_mult = (
    st.sidebar.slider("IQR multiplier", 0.5, 5.0, 1.5, 0.1) if method == "iqr" else 1.5
)
outlier_flags = (
    detect_outliers(
        df_res, y_cols, method=method, z_thresh=z_t if z_t else 3.0, iqr_mult=iqr_mult
    )
    if y_cols
    else pd.DataFrame()
)

# Control whether to exclude flagged outliers from charts
exclude_outliers_viz = st.sidebar.checkbox(
    "Exclude flagged outliers in charts", value=False, key="viz_exclude_outliers"
)
# Dataframe for visualization (optionally masked at outlier points)
df_viz = df_res.copy()
if exclude_outliers_viz and not outlier_flags.empty:
    for _c in y_cols:
        if _c in outlier_flags.columns:
            df_viz.loc[outlier_flags[_c].fillna(False), _c] = np.nan

st.sidebar.subheader("Modeling")
target_col = (
    st.sidebar.selectbox("Target", options=y_cols, index=0, key="target_column")
    if y_cols
    else None
)
test_size = st.sidebar.slider("Test size frac", 0.05, 0.5, 0.2, 0.05)
poly_degree = st.sidebar.slider("Poly degree", 2, 6, 3, 1)
skip_outliers_model = st.sidebar.checkbox("Ignore outliers while training", value=True)

TAB_OVERVIEW, TAB_TS, TAB_MODEL, TAB_REG, TAB_ENERGY, TAB_DL, TAB_HELP = st.tabs(
    [
        "Overview",
        "Time Series",
        "Modeling",
        "Regression",
        "Energy Modeling",
        "Downloads",
        "Help",
    ]
)

with TAB_OVERVIEW:
    st.subheader("Overview")
    if not len(df_res):
        st.info("Upload data to continue.")
    else:
        with st.expander("Data Preview (after cleaning & resampling)", expanded=True):
            st.dataframe(df_res, use_container_width=True)
        if not y_cols:
            st.info("Select one or more series in the sidebar to see stats.")
        else:
            rows = []
            for c in y_cols:
                vals = pd.to_numeric(df_res[c], errors="coerce")
                rows.append(
                    {
                        "Series": c,
                        "Count": int(vals.count()),
                        "Mean": vals.mean(),
                        "Std": vals.std(),
                        "Min": vals.min(),
                        "Max": vals.max(),
                        "Outliers": (
                            int(outlier_flags[c].sum())
                            if c in outlier_flags.columns
                            else 0
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if len(outlier_flags):
                st.caption("Outliers are flagged only.")

with TAB_TS:
    st.subheader("Time Series")
    if not y_cols or not len(df_res):
        st.info("Nothing to plot.")
    else:
        st.plotly_chart(
            make_line_plot(
                df_viz,
                date_col,
                y_cols,
                (
                    None
                    if exclude_outliers_viz
                    else (outlier_flags if len(outlier_flags) else None)
                ),
                apply_ma_cols,
                ma_window,
            ),
            use_container_width=True,
            key="ts_main_chart",
        )
        with st.expander("Individual series", expanded=False):
            for c in y_cols:
                one = df_viz[[date_col, c]].dropna()
                fig_ind = make_line_plot(
                    one,
                    date_col,
                    [c],
                    (
                        None
                        if exclude_outliers_viz
                        else (
                            outlier_flags[[c]] if c in outlier_flags.columns else None
                        )
                    ),
                    [c] if c in apply_ma_cols else [],
                    ma_window,
                )
                fig_ind.update_layout(
                    title=str(c),
                    xaxis_title=str(date_col),
                    yaxis_title=str(c),
                )
                st.plotly_chart(
                    fig_ind,
                    use_container_width=True,
                    key=f"ts_series_{c}",
                )
        if len(y_cols) > 1:
            corr = df_viz[y_cols].corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=True, aspect="auto", title="Correlation"),
                use_container_width=True,
                key="ts_correlation",
            )
            norm_df = df_viz[[date_col]].copy()
            for c in y_cols:
                vals = pd.to_numeric(df_viz[c], errors="coerce")
                vmin, vmax = vals.min(), vals.max()
                norm_df[c] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            fig_norm = go.Figure()
            for c in y_cols:
                fig_norm.add_trace(
                    go.Scatter(x=norm_df[date_col], y=norm_df[c], mode="lines", name=c)
                )
            fig_norm.update_layout(
                title="Normalized (Min-Max) Comparison",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_norm, use_container_width=True, key="ts_normalized")

with TAB_MODEL:
    st.subheader("Modeling")
    if not target_col or len(df_res) < 10:
        st.info("Need at least 10 rows & a target.")
    else:
        base_m = df_res.copy()
        if skip_outliers_model and target_col in outlier_flags.columns:
            base_m = base_m.loc[~outlier_flags[target_col]]
            st.caption("Ignoring flagged outliers for target.")
        X = build_time_feature(base_m, date_col)
        y = pd.to_numeric(base_m[target_col], errors="coerce").to_numpy()
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(y) < 8 or np.unique(y).size <= 1:
            st.info("Not enough variation.")
        else:
            metrics, preds, models = fit_models(
                X, y, test_size=test_size, poly_degree=poly_degree, random_state=42
            )
            st.dataframe(metrics, use_container_width=True)
            st.markdown("### Per-series details")
            for c in y_cols:
                with st.expander(c, expanded=False):
                    df_s = df_res[[date_col, c]].dropna()
                    if skip_outliers_model and c in outlier_flags.columns:
                        df_s = df_s.loc[~outlier_flags[c]]
                    y_s = pd.to_numeric(df_s[c], errors="coerce")
                    df_s = df_s.loc[~y_s.isna()]
                    if len(df_s) < 8 or df_s[c].nunique() <= 1:
                        st.caption("Insufficient data.")
                        continue
                    X_s = build_time_feature(df_s, date_col)
                    y_arr = df_s[c].to_numpy()
                    met_s, preds_s, _ = fit_models(
                        X_s,
                        y_arr,
                        test_size=test_size,
                        poly_degree=poly_degree,
                        random_state=42,
                    )
                    plot_df = df_s[[date_col]].copy()
                    plot_df[c] = y_arr
                    for col in preds_s.columns:
                        if col != "split":
                            plot_df[col] = preds_s[col].values
                    fig_s = go.Figure()
                    fig_s.add_trace(
                        go.Scatter(
                            x=plot_df[date_col],
                            y=plot_df[c],
                            mode="lines+markers",
                            name="Actual",
                        )
                    )
                    for mname in ("Linear", "Poly", "RandomForest"):
                        if mname in plot_df.columns:
                            fig_s.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[mname],
                                    mode="lines",
                                    name=mname,
                                )
                            )
                    fig_s.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified"
                    )
                    st.plotly_chart(
                        fig_s,
                        use_container_width=True,
                        key=f"model_series_{c}",
                    )
                    resid_fig = go.Figure()
                    for mname in ("Linear", "Poly", "RandomForest"):
                        if mname in plot_df.columns:
                            resid_fig.add_trace(
                                go.Scatter(
                                    x=plot_df[date_col],
                                    y=plot_df[c] - plot_df[mname],
                                    mode="markers",
                                    name=f"{mname} resid",
                                )
                            )
                    resid_fig.update_layout(
                        title="Residuals",
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(
                        resid_fig,
                        use_container_width=True,
                        key=f"model_resid_{c}",
                    )
                    st.dataframe(met_s, use_container_width=True)
            st.markdown("### Manual prediction")
            manual_date = st.date_input(
                "Prediction date",
                value=base_m[date_col].max().date(),
                min_value=base_m[date_col].min().date(),
            )
            pick_model = st.selectbox(
                "Model", metrics["Model"].tolist(), index=0, key="model_pick"
            )
            if pick_model == "Linear":
                mdl = LinearRegression()
            elif pick_model == "Poly":
                mdl = Pipeline(
                    [
                        (
                            "poly",
                            PolynomialFeatures(degree=poly_degree, include_bias=False),
                        ),
                        ("lin", LinearRegression()),
                    ]
                )
            else:
                mdl = RandomForestRegressor(n_estimators=300, random_state=42)
            mdl.fit(X, y)
            offset = (
                pd.to_datetime(manual_date) - base_m[date_col].min()
            ).total_seconds() / (24 * 3600)
            pred_target = float(mdl.predict(np.array([[offset]]))[0])
            rmse_sel = metrics.loc[metrics["Model"] == pick_model, "RMSE"].iloc[0]
            actual_series = base_m.loc[
                base_m[date_col] == pd.to_datetime(manual_date), target_col
            ]
            if not actual_series.empty:
                actual_val = float(actual_series.iloc[0])
                abs_err = abs(pred_target - actual_val)
                pct_err = abs_err / abs(actual_val) * 100 if actual_val != 0 else np.nan
                st.write(
                    f"Target {target_col} prediction {manual_date}: {pred_target:.3f} (actual {actual_val:.3f}, abs error {abs_err:.3f}, % error {pct_err:.2f}%, ±{rmse_sel:.3f} RMSE)"
                )
            else:
                st.write(
                    f"Target {target_col} prediction {manual_date}: {pred_target:.3f} (±{rmse_sel:.3f} RMSE, actual unavailable)"
                )
            rows = []
            for c in y_cols:
                df_s = df_res[[date_col, c]].dropna()
                if skip_outliers_model and c in outlier_flags.columns:
                    df_s = df_s.loc[~outlier_flags[c]]
                if len(df_s) < 5 or df_s[c].nunique() <= 1:
                    rows.append(
                        {
                            "Series": c,
                            "Prediction": np.nan,
                            "Actual": np.nan,
                            "AbsError": np.nan,
                            "PctError": np.nan,
                            "RMSE": np.nan,
                            "Note": "Insufficient",
                        }
                    )
                    continue
                base_c = df_s[date_col].min()
                X_c = (
                    ((df_s[date_col] - base_c).dt.total_seconds() / (24 * 3600))
                    .to_numpy()
                    .reshape(-1, 1)
                )
                y_c = pd.to_numeric(df_s[c], errors="coerce").to_numpy()
                ok = ~np.isnan(y_c)
                X_c, y_c = X_c[ok], y_c[ok]
                if pick_model == "Linear":
                    m_c = LinearRegression()
                elif pick_model == "Poly":
                    m_c = Pipeline(
                        [
                            (
                                "poly",
                                PolynomialFeatures(
                                    degree=poly_degree, include_bias=False
                                ),
                            ),
                            ("lin", LinearRegression()),
                        ]
                    )
                else:
                    m_c = RandomForestRegressor(n_estimators=300, random_state=42)
                try:
                    m_c.fit(X_c, y_c)
                    off = (pd.to_datetime(manual_date) - base_c).total_seconds() / (
                        24 * 3600
                    )
                    pred_c = float(m_c.predict(np.array([[off]]))[0])
                    train_pred = m_c.predict(X_c)
                    rmse_c = float(np.sqrt(np.mean((y_c - train_pred) ** 2)))
                    actual_c_series = df_s.loc[
                        df_s[date_col] == pd.to_datetime(manual_date), c
                    ]
                    if not actual_c_series.empty:
                        actual_c = float(actual_c_series.iloc[0])
                        abs_err_c = abs(pred_c - actual_c)
                        pct_err_c = (
                            abs_err_c / abs(actual_c) * 100 if actual_c != 0 else np.nan
                        )
                    else:
                        actual_c = np.nan
                        abs_err_c = np.nan
                        pct_err_c = np.nan
                    rows.append(
                        {
                            "Series": c,
                            "Prediction": pred_c,
                            "Actual": actual_c,
                            "AbsError": abs_err_c,
                            "PctError": pct_err_c,
                            "RMSE": rmse_c,
                            "Note": "",
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "Series": c,
                            "Prediction": np.nan,
                            "Actual": np.nan,
                            "AbsError": np.nan,
                            "PctError": np.nan,
                            "RMSE": np.nan,
                            "Note": f"Fail {e}",
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with TAB_REG:
    st.subheader("Regression")
    if not len(df_res) or not numeric_cols:
        st.info("Upload data with numeric columns.")
    else:
        dep_col = st.selectbox(
            "Dependent variable", options=numeric_cols, index=0, key="dep_variable"
        )
        indep_opts = [c for c in numeric_cols if c != dep_col]
        indep_cols = st.multiselect(
            "Independent variables", options=indep_opts, default=indep_opts[:1]
        )
        if not indep_cols:
            st.info("Select at least one independent variable.")
        else:
            df_reg = df_res[[dep_col] + indep_cols].dropna()
            # Optional filters to remove unwanted regions (e.g., left-side points)
            with st.expander(
                "Filters (Regression) — adjust ranges to exclude outliers or left clusters",
                expanded=True,
            ):
                filters = {}
                base_df = df_reg.copy()
                for c in [dep_col] + indep_cols:
                    vals = pd.to_numeric(base_df[c], errors="coerce")
                    lo, hi = float(vals.min()), float(vals.max())
                    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                        step = (hi - lo) / 100.0 if (hi - lo) > 0 else 1.0
                        sel = st.slider(
                            f"{c} range",
                            min_value=lo,
                            max_value=hi,
                            value=(lo, hi),
                            step=step,
                            key=f"reg_range_{c}",
                        )
                        filters[c] = sel
                # Apply filters
                orig_n = len(df_reg)
                for c, (lo, hi) in filters.items():
                    v = pd.to_numeric(df_reg[c], errors="coerce")
                    df_reg = df_reg[(v >= lo) & (v <= hi)]
                if len(df_reg) != orig_n:
                    st.caption(
                        f"Filtered rows: {orig_n - len(df_reg)} removed, {len(df_reg)} kept."
                    )
            if len(df_reg) < 5:
                st.info("Need at least 5 rows after dropping NA.")
            else:
                X = df_reg[indep_cols].to_numpy()
                y = df_reg[dep_col].to_numpy()
                metrics_r, preds_r, models_r = fit_models(
                    X, y, test_size=test_size, poly_degree=poly_degree, random_state=42
                )
                st.dataframe(metrics_r, use_container_width=True)
                st.plotly_chart(
                    px.imshow(
                        df_reg.corr(),
                        text_auto=True,
                        aspect="auto",
                        title="Correlation",
                    ),
                    use_container_width=True,
                    key="reg_correlation",
                )
                for c in indep_cols:
                    st.plotly_chart(
                        px.scatter(
                            df_reg,
                            x=c,
                            y=dep_col,
                            trendline="ols",
                            title=f"{c} vs {dep_col}",
                        ),
                        use_container_width=True,
                        key=f"reg_scatter_{c}",
                    )
                with st.expander("Manual prediction", expanded=False):
                    mdl_name = st.selectbox(
                        "Model", metrics_r["Model"].tolist(), index=0, key="reg_model"
                    )
                    inputs = []
                    for c in indep_cols:
                        inputs.append(
                            st.number_input(
                                c,
                                value=float(df_reg[c].mean()),
                                key=f"{c}_input_reg",
                            )
                        )
                    if st.button("Predict", key="predict_reg"):
                        mdl = models_r[mdl_name]
                        pred = float(mdl.predict(np.array([inputs]))[0])
                        rmse_sel = metrics_r.loc[
                            metrics_r["Model"] == mdl_name, "RMSE"
                        ].iloc[0]
                        df_match = df_reg.copy()
                        for col, val in zip(indep_cols, inputs):
                            df_match = df_match.loc[np.isclose(df_match[col], val)]
                        if len(df_match):
                            actual_val = float(df_match[dep_col].iloc[0])
                            abs_err = abs(pred - actual_val)
                            pct_err = (
                                abs_err / abs(actual_val) * 100
                                if actual_val != 0
                                else np.nan
                            )
                            st.write(
                                f"Predicted {dep_col}: {pred:.3f} (actual {actual_val:.3f}, abs error {abs_err:.3f}, % error {pct_err:.2f}%, ±{rmse_sel:.3f} RMSE)"
                            )
                        else:
                            st.write(
                                f"Predicted {dep_col}: {pred:.3f} (±{rmse_sel:.3f} RMSE, actual unavailable)"
                            )

with TAB_ENERGY:
    st.subheader("Energy Modeling")
    st.markdown(
        "Use monthly degassing station data to model fuel gas and electricity consumption and intensities."
    )

    # Data & Features
    st.markdown("### Data & Features")
    c1, c2 = st.columns([2, 1])
    with c1:
        energy_file = st.file_uploader(
            "Upload monthly CSV/XLSX", type=["csv", "xlsx"], key="energy_upload"
        )
    with c2:
        if st.button("Use example data", key="energy_example"):
            example = pd.DataFrame(
                {
                    "date": pd.date_range("2022-01-01", periods=24, freq="MS"),
                    "crude": np.random.uniform(100, 200, 24).round(1),
                    "water": np.random.uniform(50, 120, 24).round(1),
                    "gas": np.random.uniform(10, 30, 24).round(1),
                    "amb_temp": np.random.uniform(20, 40, 24).round(1),
                    "fuel_gas": np.random.uniform(3, 8, 24).round(2),
                    "electricity": np.random.uniform(1500, 3000, 24).round(0),
                }
            )
            st.session_state["energy_df_raw"] = example
    base_temp = st.selectbox(
        "Base temperature for CDD (°C)",
        options=[20.0, 22.0, 24.0, 25.0, 26.0, 28.0, 30.0],
        index=3,
        key="energy_base",
    )

    # Read upload
    energy_df_raw = st.session_state.get("energy_df_raw")
    if energy_file is not None:
        if energy_file.name.lower().endswith(".csv"):
            energy_df_raw = pd.read_csv(energy_file)
        else:
            energy_df_raw = pd.read_excel(energy_file)
        st.session_state["energy_df_raw"] = energy_df_raw

    if energy_df_raw is None:
        st.info("Upload a file or use the example data.")
    else:
        # Sidebar controls
        with st.sidebar.expander("Energy Data: Cleaning & Features", expanded=True):
            mv = st.selectbox(
                "Missing values",
                ["none", "ffill", "bfill", "interp"],
                index=0,
                key="energy_mv",
            )
            out_m = st.selectbox(
                "Outliers method", ["zscore", "iqr"], index=1, key="energy_out_m"
            )
            z_thr = (
                st.slider("Z threshold", 2.0, 5.0, 3.0, 0.1, key="energy_z")
                if out_m == "zscore"
                else None
            )
            iqr_thr = (
                st.slider("IQR multiplier", 0.5, 5.0, 1.5, 0.1, key="energy_iqr")
                if out_m == "iqr"
                else 1.5
            )

        # Prepare
        edf = energy_df_raw.copy()
        edf.columns = [str(c).strip().lower() for c in edf.columns]
        if "date" not in edf.columns:
            st.error("Energy data needs a 'date' column.")
        else:
            edf["date"] = pd.to_datetime(edf["date"], errors="coerce")
            edf = edf.dropna(subset=["date"]).sort_values("date")
            edf = edf.set_index("date").asfreq("MS").reset_index()

            if mv != "none":
                num_cols = [c for c in edf.columns if c != "date"]
                if mv == "ffill":
                    edf[num_cols] = edf[num_cols].ffill()
                elif mv == "bfill":
                    edf[num_cols] = edf[num_cols].bfill()
                else:
                    edf[num_cols] = edf[num_cols].interpolate(limit_direction="both")

            # Outlier flags on core numeric columns
            core_cols = [
                c
                for c in [
                    "crude",
                    "water",
                    "gas",
                    "amb_temp",
                    "fuel_gas",
                    "electricity",
                ]
                if c in edf.columns
            ]
            flags_energy = detect_outliers(
                edf,
                core_cols,
                method=out_m,
                z_thresh=z_thr if z_thr else 3.0,
                iqr_mult=iqr_thr,
            )

            # Features
            edf_feat = add_features(edf, base_temp=float(base_temp))
            st.dataframe(edf_feat.tail(12), use_container_width=True)

            # Feature/Target selection (Sidebar)
            with st.sidebar.expander(
                "Energy Modeling: Select target & features", expanded=True
            ):
                all_features = [
                    c
                    for c in edf_feat.columns
                    if c
                    not in [
                        "date",
                        "fuel_gas",
                        "electricity",
                        "fg_intensity",
                        "el_intensity",
                    ]
                ]
                target = st.selectbox(
                    "Target",
                    options=["fuel_gas", "electricity", "fg_intensity", "el_intensity"],
                    index=0,
                    key="energy_target",
                )
                sel_feats = st.multiselect(
                    "Features",
                    options=all_features,
                    default=[
                        c
                        for c in [
                            "total_liquid",
                            "CDD",
                            "crude_rm3",
                            "gas_rm3",
                            "amb_temp_rm3",
                        ]
                        if c in all_features
                    ],
                    key="energy_feats",
                )

            # Model comparison
            st.markdown("### Model Comparison")
            metrics_e, fitted_e, preds_e = train_compare(
                edf_feat, target=target, feature_cols=sel_feats
            )
            if metrics_e.empty:
                st.info("Not enough data after cleaning/feature selection.")
            else:
                # Highlight best (smallest RMSE)
                best_row = metrics_e.iloc[metrics_e["RMSE"].astype(float).argmin()]
                best_name = str(best_row["Model"])
                st.dataframe(
                    metrics_e.style.highlight_min(subset=["RMSE"], color="#d1ffd1"),
                    use_container_width=True,
                )

                # Actual vs Predicted
                st.markdown("### Actual vs Predicted")
                if not preds_e.empty:
                    fig_ap = go.Figure()
                    x_vals = (
                        edf_feat["date"]
                        if "date" in edf_feat.columns
                        else np.arange(len(preds_e))
                    )
                    fig_ap.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=edf_feat[target],
                            mode="lines+markers",
                            name="Actual",
                        )
                    )
                    if best_name in preds_e.columns:
                        fig_ap.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=preds_e[best_name],
                                mode="lines",
                                name=f"Predicted ({best_name})",
                            )
                        )
                    fig_ap.update_layout(
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_title="date",
                        yaxis_title=target,
                    )
                    st.plotly_chart(
                        fig_ap, use_container_width=True, key="energy_actual_pred"
                    )

                # Importance / coefficients
                if best_name in fitted_e:
                    best_model = fitted_e[best_name]
                    st.markdown("#### Feature importance / coefficients")
                    imp_vals = None
                    try:
                        # Tree-based
                        if hasattr(best_model, "feature_importances_"):
                            imp_vals = np.array(best_model.feature_importances_)
                        elif hasattr(best_model, "named_steps"):
                            last = best_model.named_steps.get("model")
                            if last is not None and hasattr(last, "coef_"):
                                coef = np.ravel(last.coef_)
                                imp_vals = np.abs(coef)
                        elif hasattr(best_model, "coef_"):
                            coef = np.ravel(best_model.coef_)
                            imp_vals = np.abs(coef)
                    except Exception:
                        imp_vals = None

                    if imp_vals is not None and len(imp_vals) == len(sel_feats):
                        imp_df = pd.DataFrame(
                            {"feature": sel_feats, "importance": imp_vals}
                        ).sort_values("importance", ascending=False)
                        st.plotly_chart(
                            px.bar(
                                imp_df,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title=f"{best_name} importance",
                            ),
                            use_container_width=True,
                            key="energy_importance",
                        )
                    else:
                        st.caption("Importance not available for the best model.")

                # What-If Analysis
                st.markdown("### What-If Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    # crude slider with safe bounds
                    if "crude" in edf_feat:
                        s = pd.to_numeric(edf_feat["crude"], errors="coerce")
                        _max = (
                            float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else np.nan
                        )
                        _med = (
                            float(np.nanmedian(s))
                            if np.isfinite(np.nanmedian(s))
                            else np.nan
                        )
                    else:
                        _max, _med = np.nan, np.nan
                    lo, hi = 0.0, (_max * 1.5 if np.isfinite(_max) else 300.0)
                    if not np.isfinite(hi) or hi <= lo:
                        hi = 300.0
                    default = _med if np.isfinite(_med) else 150.0
                    default = min(max(default, lo), hi)
                    step = max((hi - lo) / 100.0, 0.01)
                    crude_in = st.slider(
                        "crude", lo, hi, default, step=step, key="energy_crude"
                    )

                    # water slider with safe bounds
                    if "water" in edf_feat:
                        s = pd.to_numeric(edf_feat["water"], errors="coerce")
                        _max = (
                            float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else np.nan
                        )
                        _med = (
                            float(np.nanmedian(s))
                            if np.isfinite(np.nanmedian(s))
                            else np.nan
                        )
                    else:
                        _max, _med = np.nan, np.nan
                    lo, hi = 0.0, (_max * 1.5 if np.isfinite(_max) else 300.0)
                    if not np.isfinite(hi) or hi <= lo:
                        hi = 300.0
                    default = _med if np.isfinite(_med) else 100.0
                    default = min(max(default, lo), hi)
                    step = max((hi - lo) / 100.0, 0.01)
                    water_in = st.slider(
                        "water", lo, hi, default, step=step, key="energy_water"
                    )

                    # gas slider with safe bounds
                    if "gas" in edf_feat:
                        s = pd.to_numeric(edf_feat["gas"], errors="coerce")
                        _max = (
                            float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else np.nan
                        )
                        _med = (
                            float(np.nanmedian(s))
                            if np.isfinite(np.nanmedian(s))
                            else np.nan
                        )
                    else:
                        _max, _med = np.nan, np.nan
                    lo, hi = 0.0, (_max * 1.5 if np.isfinite(_max) else 100.0)
                    if not np.isfinite(hi) or hi <= lo:
                        hi = 100.0
                    default = _med if np.isfinite(_med) else 20.0
                    default = min(max(default, lo), hi)
                    step = max((hi - lo) / 100.0, 0.01)
                    gas_in = st.slider(
                        "gas", lo, hi, default, step=step, key="energy_gas"
                    )
                    amb_in = st.slider("amb_temp", 0.0, 50.0, 30.0, key="energy_amb")
                with c2:
                    wc_override = st.checkbox(
                        "Override water_cut", value=False, key="energy_wc_ov"
                    )
                    wc_val = (
                        st.slider(
                            "water_cut (0-1)", 0.0, 1.0, 0.3, 0.01, key="energy_wc"
                        )
                        if wc_override
                        else None
                    )
                    gor_override = st.checkbox(
                        "Override GOR", value=False, key="energy_gor_ov"
                    )
                    if gor_override:
                        if "gor" in edf_feat:
                            s = pd.to_numeric(edf_feat["gor"], errors="coerce")
                            gmax = (
                                float(np.nanmax(s))
                                if np.isfinite(np.nanmax(s))
                                else np.nan
                            )
                        else:
                            gmax = np.nan
                        lo, hi = 0.0, (
                            float(max(1.0, gmax)) if np.isfinite(gmax) else 10.0
                        )
                        if not np.isfinite(hi) or hi <= lo:
                            hi = 10.0
                        step = max((hi - lo) / 100.0, 0.01)
                        gor_val = st.slider(
                            "GOR",
                            lo,
                            hi,
                            min(max(2.0, lo), hi),
                            step=step,
                            key="energy_gor",
                        )
                    else:
                        gor_val = None

                scenario = pd.DataFrame(
                    {
                        "date": [pd.Timestamp.today().replace(day=1)],
                        "crude": [crude_in],
                        "water": [water_in],
                        "gas": [gas_in],
                        "amb_temp": [amb_in],
                        "fuel_gas": [0.0],
                        "electricity": [0.0],
                    }
                )
                scen_feat = add_features(scenario, base_temp=float(base_temp))
                if wc_override:
                    scen_feat["water_cut"] = wc_val
                if gor_override and gor_val is not None:
                    scen_feat["gor"] = gor_val

                # Predict both fuel_gas and electricity using best-by-RMSE models for each target separately
                preds_out = {}
                for tgt in ["fuel_gas", "electricity", "fg_intensity", "el_intensity"]:
                    met_t, fit_t, _ = train_compare(
                        edf_feat, target=tgt, feature_cols=sel_feats
                    )
                    if not met_t.empty:
                        best_t = met_t.iloc[met_t["RMSE"].astype(float).argmin()][
                            "Model"
                        ]
                        model_t = fit_t.get(str(best_t))
                        if model_t is not None:
                            Xs = scen_feat[sel_feats].to_numpy()
                            try:
                                preds_out[tgt] = float(model_t.predict(Xs)[0])
                            except Exception:
                                preds_out[tgt] = np.nan
                st.write(
                    {
                        k: round(v, 3) if v is not None and np.isfinite(v) else v
                        for k, v in preds_out.items()
                    }
                )

                # Downloads
                st.markdown("### Downloads")
                st.download_button(
                    "⬇️ Data with features (CSV)",
                    edf_feat.to_csv(index=False).encode("utf-8"),
                    file_name="energy_features.csv",
                )
                st.download_button(
                    "⬇️ Metrics (CSV)",
                    metrics_e.to_csv(index=False).encode("utf-8"),
                    file_name="energy_metrics.csv",
                )
                if not preds_e.empty:
                    pred_csv = (
                        pd.concat([edf_feat[["date", target]], preds_e], axis=1)
                        .to_csv(index=False)
                        .encode("utf-8")
                    )
                    st.download_button(
                        "⬇️ Predictions (CSV)",
                        pred_csv,
                        file_name="energy_predictions.csv",
                    )
with TAB_DL:
    st.subheader("Downloads")
    cleaned_csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Cleaned data (CSV)",
        cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
    if len(outlier_flags):
        st.download_button(
            "⬇️ Outlier flags (CSV)",
            outlier_flags.astype(int).to_csv(index=False).encode("utf-8"),
            file_name="outliers_flags.csv",
            mime="text/csv",
        )
    if "metrics" in locals():
        st.download_button(
            "⬇️ Target model metrics (CSV)",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name="model_metrics.csv",
            mime="text/csv",
        )

with TAB_HELP:
    st.subheader("Help & Guide")
    st.markdown(
        """### 1. Upload
Provide CSV with a date column + numeric columns.

### 2. Clean
Optionally fill gaps (forward/backward/interpolate).

### 3. Outliers
Flag unusual points (Z-score or IQR). They are not removed unless you ignore them in modeling.

### 4. Charts
Main time series, individual series, correlations, normalized comparison.

### 5. Modeling
Simple time‑trend using Linear / Polynomial / RandomForest. Manual prediction lets you pick a date and model.

### 6. Downloads
Export cleaned data, flags, metrics.
"""
    )
