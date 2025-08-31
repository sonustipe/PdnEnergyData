import io, csv, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Energy & Operations - Correlations & Regression", layout="wide")

st.title("Energy & Operations - Correlations & Regression (Streamlit)")

# ----------------------
# Sidebar: Configuration
# ----------------------
st.sidebar.header("Configuration")

st.sidebar.markdown("**Energy conversion factors** (edit if needed):")
ELECTRICITY_MWH_TO_GJ = st.sidebar.number_input("1 MWh = GJ", value=3.6, min_value=0.1, step=0.1, format="%.1f")
FUEL_MMSCF_TO_GJ = st.sidebar.number_input("1 MMSCF Gas = GJ (HHV)", value=1055.0, min_value=1.0, step=1.0, format="%.1f")

st.sidebar.markdown("---")
dayfirst = st.sidebar.checkbox("Dates are DD-MM-YYYY", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Polynomial degree** (for model diagnostics):")
poly_degree = st.sidebar.slider("Degree", min_value=1, max_value=4, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Select features** for multivariate regression")

# ----------------------
# Helper: robust CSV reader to fix split thousands in numeric fields
# ----------------------
def read_dirty_csv(text):
    rows = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows.append(header)

    for row in reader:
        fixed = []
        i = 0
        while i < len(row):
            token = row[i]
            # Heuristic: fix thousands split like ["1","834"] -> "1834"
            if i + 1 < len(row) and token.isdigit() and row[i+1].isdigit() and len(row[i+1]) == 3:
                token = token + row[i+1]
                i += 2
                fixed.append(token)
                continue
            fixed.append(token)
            i += 1

        # If still too many columns, collapse extras into the last numeric field(s)
        while len(fixed) > len(header):
            fixed[-2] = fixed[-2] + fixed.pop(-1)
        rows.append(fixed)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

# ----------------------
# File uploader or sample
# ----------------------
st.subheader("1) Load your CSV")
uploaded = st.file_uploader("Upload your monthly CSV (columns: date, crude oil, h2o, gas part, Fuel, Electricity, Temperature). Unquoted thousands like 1,834 are OK.", type=["csv"])

sample_csv = """date,crude oil,h2o,gas part,Fuel,Electricity,Temperature
01-01-2024,27106.93548,38436.74194,61.12903226,2.474662715,1652.87,21.13870968
01-02-2024,27851.28571,41114.25,64.96428571,2.463161391,1730.01,20.65
01-03-2024,26972.58065,40069.80645,67.5483871,2.407725439,1845.29,22.80645161
01-04-2024,27844.3,39993.03333,62.93333333,2.136002786,1,834,27.85666667
01-05-2024,28044.06452,37916.77419,73.38709677,2.188694464,1933.93,33.09354839
01-06-2024,24787.9,35198.86667,62.13333333,2.117026791,1,950,36.86333333
01-07-2024,25558.16129,44198.51613,67,1.997079668,2,235,38.68709677
01-08-2024,26126.87097,42557.45161,59.77419355,2.166671865,2,354,36.49354839
01-09-2024,25422.6,40442.43333,57.26666667,2.169182742,1974.336,34.80333333
01-10-2024,23667.06452,40761.83871,48.4516129,1.886541591,1959.864,31.32903226
01-11-2024,25901.13333,41427,49.83333333,2.146264508,1527.993,26.64333333
01-12-2024,26776.45161,43995.32258,47.67741935,2.296204424,2036.047,20.52580645
"""

if uploaded is not None:
    raw_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    df0 = read_dirty_csv(raw_text)
else:
    st.info("No file uploaded. Using the bundled sample data so you can try the app.")
    df0 = read_dirty_csv(sample_csv)

# Normalize column names
df0.columns = [c.strip().lower().replace(" ", "_") for c in df0.columns]
expected_cols = ['date','crude_oil','h2o','gas_part','fuel','electricity','temperature']
missing = [c for c in expected_cols if c not in df0.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Type parsing
df0['date'] = pd.to_datetime(df0['date'], dayfirst=dayfirst, errors='coerce')
for col in ['crude_oil','h2o','gas_part','fuel','electricity','temperature']:
    df0[col] = pd.to_numeric(df0[col].astype(str).str.replace(',', ''), errors='coerce')

df0 = df0.dropna(subset=['date']).reset_index(drop=True)

st.write("### Raw (parsed) data")
st.dataframe(df0)

# ----------------------
# Feature engineering
# ----------------------
df = df0.copy()
df['fuel_energy_GJ']  = df['fuel'] * FUEL_MMSCF_TO_GJ
df['fuel_energy_MWh'] = df['fuel_energy_GJ'] / ELECTRICITY_MWH_TO_GJ
df['elec_energy_MWh'] = df['electricity']
df['total_energy_MWh'] = df['fuel_energy_MWh'] + df['elec_energy_MWh']

df['total_fluids'] = df['crude_oil'] + df['h2o']
df['water_cut_pct'] = 100.0 * df['h2o'] / df['total_fluids']
df['gas_share'] = df['gas_part'] / 100.0

df['energy_per_crude_MWh_per_unit']  = df['total_energy_MWh'] / df['crude_oil']
df['energy_per_fluids_MWh_per_unit'] = df['total_energy_MWh'] / df['total_fluids']
df['gas_to_liquids_ratio'] = df['gas_share'] / (1 - df['gas_share']).replace(0, np.nan)
df['energy_to_gas_share']  = df['total_energy_MWh'] / df['gas_share'].replace(0, np.nan)

st.write("### Engineered columns preview")
st.dataframe(df[['date','fuel_energy_MWh','elec_energy_MWh','total_energy_MWh','water_cut_pct','energy_per_crude_MWh_per_unit']])

# ----------------------
# Correlation heatmap
# ----------------------
st.subheader("2) Correlation Matrix (Pearson)")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols].corr(method='pearson')

fig_corr = px.imshow(
    corr,
    x=num_cols,
    y=num_cols,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Correlation Heatmap (Pearson)",
)
fig_corr.update_layout(margin=dict(l=40, r=20, t=60, b=40))
fig_corr.update_xaxes(tickangle=45)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("**Target for correlation ranking**")
target_for_corr = st.selectbox(
    "Pick a target to rank correlations",
    options=num_cols,
    index=num_cols.index('total_energy_MWh') if 'total_energy_MWh' in num_cols else 0,
)
ranked = corr[target_for_corr].sort_values(ascending=False).to_frame("corr_with_target")
st.write(ranked)

# ----------------------
# Univariate scatter + regression line
# ----------------------
st.subheader("3) Univariate Regression Diagnostics")
target = st.selectbox(
    "Dependent variable (Y)",
    options=['total_energy_MWh','energy_per_crude_MWh_per_unit','energy_per_fluids_MWh_per_unit']
)
cand_features = [c for c in num_cols if c != target]
picked = st.multiselect(
    "Pick X features to plot (univariate diagnostics)",
    cand_features,
    default=['crude_oil','h2o','gas_part','temperature']
)

deg = st.slider("Polynomial degree for univariate fit", min_value=1, max_value=4, value=1, step=1)

for xcol in picked:
    x = df[[xcol]].values
    y = df[target].values
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('lin', LinearRegression()),
    ])
    model.fit(x, y)

    # LOOCV R^2
    try:
        loo = LeaveOneOut()
        loocv_scores = cross_val_score(model, x, y, cv=loo, scoring='r2')
        r2_loocv = np.nanmean(loocv_scores)
    except Exception:
        r2_loocv = np.nan

    r2_fit = model.score(x, y)

    # Prepare prediction curve
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200).reshape(-1, 1)
    ys = model.predict(xs)

    # Plotly scatter + regression line
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='Data'))
    fig_scatter.add_trace(go.Scatter(x=xs.flatten(), y=ys, mode='lines', name=f'Poly deg {deg}'))
    fig_scatter.update_layout(
        title=f"{xcol} vs {target} | R^2 fit={r2_fit:.3f} | R^2 LOOCV={r2_loocv:.3f}",
        xaxis_title=xcol,
        yaxis_title=target,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------
# Multivariate regression summary
# ----------------------
st.subheader("4) Multivariate Regression (summary)")
multifeats = st.multiselect(
    "Select X features for the multivariate model",
    cand_features,
    default=['crude_oil','h2o','gas_part','temperature','water_cut_pct']
)

if multifeats:
    X = df[multifeats].values
    y = df[target].values

    deg_multi = st.slider("Polynomial degree for multivariate model", min_value=1, max_value=3, value=2, step=1)

    model_multi = Pipeline([
        ('poly', PolynomialFeatures(degree=deg_multi, include_bias=False)),
        ('lin', LinearRegression()),
    ])
    model_multi.fit(X, y)

    r2_fit_m = model_multi.score(X, y)
    try:
        loo = LeaveOneOut()
        r2_loocv_m = np.nanmean(cross_val_score(model_multi, X, y, cv=loo, scoring='r2'))
    except Exception:
        r2_loocv_m = np.nan

    st.write(pd.DataFrame({
        "model":[f"Poly degree {deg_multi}"],
        "R2_fit_all":[r2_fit_m],
        "R2_LOOCV":[r2_loocv_m],
        "features":[[f for f in multifeats]],
    }))

# ----------------------
# Time series
# ----------------------
st.subheader("5) Time Series")
series_cols = ['crude_oil','h2o','gas_part','temperature','fuel_energy_MWh','elec_energy_MWh','total_energy_MWh','energy_per_crude_MWh_per_unit']
sel_series = st.multiselect("Pick series to plot over time", series_cols, default=['total_energy_MWh','energy_per_crude_MWh_per_unit'])

if sel_series:
    for col in sel_series:
        fig_ts = px.line(df, x='date', y=col, title=col)
        fig_ts.update_traces(mode='lines+markers')
        fig_ts.update_layout(margin=dict(l=40, r=20, t=60, b=40), xaxis_title='Date', yaxis_title=col)
        st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")
st.caption("Charts use Plotly for interactivity. Switch targets, features, and polynomial degrees for diagnostics. Upload your CSV to replace the sample.")

