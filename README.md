# Time Series Tab Guide (Preliminary Energy Modelling)

This repository includes a Streamlit dashboard for energy and operations diagnostics.
The **Time Series** tab (`5) Time Series`) is designed to quickly show how key process and energy indicators evolve month-to-month, and to support **early-stage (preliminary) energy modelling** decisions.

## What the Time Series tab does

In `app_streamlit_energy.py`, the Time Series section:

- lets you choose one or more variables from a predefined list,
- draws an individual interactive line chart for each selected variable,
- plots `date` on the x-axis and the selected metric on the y-axis,
- uses **lines + markers** so both trend and individual data points are visible.

This gives a fast visual check of trajectory, volatility, and possible operational shifts before moving into more formal regression modelling.

## Graphs available in the Time Series tab

The selectable series are:

1. `crude_oil`  
   Crude production trend.
2. `h2o`  
   Produced water trend.
3. `gas_part`  
   Gas fraction/percentage trend.
4. `temperature`  
   Ambient/process temperature trend.
5. `fuel_energy_MWh`  
   Fuel energy converted to MWh.
6. `elec_energy_MWh`  
   Electrical energy usage (MWh).
7. `total_energy_MWh`  
   Combined fuel + electricity energy demand.
8. `energy_per_crude_MWh_per_unit`  
   Energy intensity per crude unit (efficiency-style KPI).

## Methods behind the Time Series plots

Although the tab is intentionally simple, it relies on consistent preprocessing and feature engineering done earlier in the app.

### 1) Data parsing and cleaning

- Dates are parsed to proper datetime values.
- Numeric fields are sanitized (including comma-separated thousands).
- Invalid dates are dropped.

Why this matters: trend analysis can be misleading if time ordering or numeric conversion is incorrect.

### 2) Energy normalization/conversion

The app converts fuel to energy and combines it with electricity:

- `fuel_energy_GJ = fuel × FUEL_MMSCF_TO_GJ`
- `fuel_energy_MWh = fuel_energy_GJ / ELECTRICITY_MWH_TO_GJ`
- `total_energy_MWh = fuel_energy_MWh + elec_energy_MWh`

Why this matters: preliminary energy modelling needs all energy inputs on a common basis before comparing demand against production behavior.

### 3) Derived KPI construction

The tab includes direct process variables and derived KPIs such as:

- `energy_per_crude_MWh_per_unit`

Why this matters: absolute energy can rise simply because throughput rises; intensity KPIs help indicate whether efficiency is improving or degrading.

### 4) Interactive visual diagnostics

Each selected metric is shown as a Plotly line chart with markers.

Why this matters: you can quickly detect:

- direction (uptrend/downtrend),
- seasonality hints,
- spikes/outliers,
- possible regime changes,
- timing alignment between process and energy variables.

## Significance for preliminary energy modelling

The Time Series tab is a **first-pass diagnostic layer**. It is not the final predictive model; it helps shape modelling choices.

### A) Variable screening

Before fitting regressions, use trends to decide which variables are plausible drivers (for example, whether `total_energy_MWh` appears to co-move with `h2o` or `crude_oil`).

### B) Data quality and instrumentation checks

Sudden jumps, flat-lines, or discontinuities often indicate meter issues, reporting changes, or operating mode shifts. Catching these early prevents biased model coefficients.

### C) Baseline and drift identification

The charts help define a baseline operating period and reveal drift over time. This is useful when deciding whether one global model is adequate or segmented models are needed.

### D) Intensity-based performance interpretation

Tracking `energy_per_crude_MWh_per_unit` helps separate throughput effects from efficiency effects. This is critical in early benchmarking and opportunity identification.

### E) Informing next modelling steps

The visual findings from this tab should be used to guide:

- correlation interpretation,
- univariate diagnostics,
- multivariate feature selection,
- degree/complexity choices for polynomial models.

## Practical usage workflow

1. Start with `total_energy_MWh` and `energy_per_crude_MWh_per_unit`.
2. Add `crude_oil`, `h2o`, and `gas_part` to inspect co-movement.
3. Add `temperature` to check weather-driven effects.
4. Flag unusual months before finalizing regression training windows.
5. Move to correlation and regression tabs with hypotheses from observed temporal behavior.

## Important limitations

- Time series lines are descriptive, not causal.
- Monthly data may hide intra-month variability.
- Small samples can overemphasize random variation.
- Visual co-movement should be confirmed with cross-validation and engineering context.

---

If useful, this README can be expanded with examples of interpretation patterns (e.g., rising water cut with rising specific energy) and recommended thresholds/alerts for operations teams.
