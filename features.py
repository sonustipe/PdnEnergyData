from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d_safe = d.replace(0, np.nan)
    out = n / d_safe
    return out.fillna(0.0)


@st.cache_data(show_spinner=False)
def add_features(df: pd.DataFrame, base_temp: float = 25.0) -> pd.DataFrame:
    """Engineer features for energy modeling.

    Inputs
    - df: requires columns ['date','crude','water','gas','amb_temp','fuel_gas','electricity'] (case-insensitive)
    - base_temp: base temperature for Cooling Degree Days (CDD)

    Returns a new DataFrame with monthly start frequency (MS) and engineered columns:
    - total_liquid, water_cut, gor, fg_intensity, el_intensity, CDD
    - rolling means (3M) for crude, water, gas, amb_temp, total_liquid
    - lags (1) for crude, water, gas, amb_temp, total_liquid
    - interactions: crude_x_CDD, gas_x_CDD, wc_x_CDD
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    # Normalize column names
    data.columns = [str(c).strip().lower() for c in data.columns]

    if "date" not in data.columns:
        # try to find a date-like column
        date_like = [c for c in data.columns if "date" in c]
        if not date_like:
            raise ValueError("No 'date' column found for Energy Modeling.")
        data.rename(columns={date_like[0]: "date"}, inplace=True)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])  # drop rows without dates
    data = data.sort_values("date")
    # Set monthly frequency at month start
    data = data.set_index("date").asfreq("MS").reset_index()

    # Ensure numeric types
    for c in ["crude", "water", "gas", "amb_temp", "fuel_gas", "electricity"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        else:
            # Missing columns default to 0 to keep the pipeline robust
            data[c] = 0.0

    # Core engineered features
    data["total_liquid"] = data["crude"].fillna(0) + data["water"].fillna(0)
    data["water_cut"] = _safe_div(data["water"], data["total_liquid"]).clip(0, 1)
    data["gor"] = _safe_div(data["gas"], data["crude"]).clip(lower=0)
    data["fg_intensity"] = _safe_div(data["fuel_gas"], data["total_liquid"]).clip(
        lower=0
    )
    data["el_intensity"] = _safe_div(data["electricity"], data["total_liquid"]).clip(
        lower=0
    )
    data["CDD"] = (data["amb_temp"].fillna(0) - float(base_temp)).clip(lower=0)

    # Rolling means (3 months)
    for c in ["crude", "water", "gas", "amb_temp", "total_liquid"]:
        data[f"{c}_rm3"] = (
            pd.to_numeric(data[c], errors="coerce").rolling(3, min_periods=1).mean()
        )

    # Lags (1 month)
    for c in ["crude", "water", "gas", "amb_temp", "total_liquid"]:
        data[f"{c}_lag1"] = pd.to_numeric(data[c], errors="coerce").shift(1)

    # Interactions
    data["crude_x_CDD"] = data["crude"].fillna(0) * data["CDD"].fillna(0)
    data["gas_x_CDD"] = data["gas"].fillna(0) * data["CDD"].fillna(0)
    data["wc_x_CDD"] = data["water_cut"].fillna(0) * data["CDD"].fillna(0)

    # Fill engineered NA from lags with 0 to simplify modeling; keep targets as is
    eng_cols = [
        "total_liquid",
        "water_cut",
        "gor",
        "fg_intensity",
        "el_intensity",
        "CDD",
        "crude_rm3",
        "water_rm3",
        "gas_rm3",
        "amb_temp_rm3",
        "total_liquid_rm3",
        "crude_lag1",
        "water_lag1",
        "gas_lag1",
        "amb_temp_lag1",
        "total_liquid_lag1",
        "crude_x_CDD",
        "gas_x_CDD",
        "wc_x_CDD",
    ]
    for c in eng_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0)

    return data
