# NPL-Early-Warning-Model

## Overview

This project builds a two-level Early Warning System (EWS) for predicting changes in Non-Performing Loans (NPL) in the Russian corporate credit portfolio using publicly available macroeconomic indicators.

**Research question:** Can macroeconomic indicators provide a reliable early warning signal for corporate credit portfolio deterioration under conditions of a short sample (80 observations) and three structurally different shocks?

**Context:** The observation period (2019–2025) includes three qualitatively different shocks — the COVID-19 pandemic, the February 2022 sanctions shock, and the high interest rate cycle of 2023–2025 (key rate reaching 21%). This creates a unique empirical setting for testing model robustness under structural breaks.

## Key Findings

### 1. Horizon Effect (Main Finding)

Macroeconomic models only outperform naive forecasts at long horizons. At H=3 months, all models perform worse than the naive baseline. At H=12 months, Ridge achieves R² = 0.87.

```
R² by Horizon:
H=3:   Naive=-0.96  AR(1)=-0.02  OLS=-1.44  Lasso=-0.33  Ridge=-0.14  ← all fail
H=6:   Naive=-2.51  AR(1)=+0.39  OLS=-0.26  Lasso=+0.14  Ridge=-0.28
H=12:  Naive=-0.85  AR(1)=+0.78  OLS=+0.96  Lasso=+0.06  Ridge=+0.87  ← macro works
```

**Implication:** EWS based on macro indicators requires a 12-month horizon. Standard EWS practice of 3–6 month horizons is suboptimal for Russian data.

### 2. Structural Break After February 2022 Sanctions

Formal tests confirm a statistically significant regime shift:

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Chow test (full model, 18 params) | F = 1.25 | 0.31 | Not significant — low power |
| Chow test (top-5 SHAP variables) | F = 5.22 | < 0.001 | ✅ Significant at 0.1% level |
| Interaction test | 6 variables | p < 0.10 | ✅ 4 sign changes |

## 3. Complete Rotation of Top-5 Predictors Between Regimes

| Rank | Pre-sanctions (2019–2022) | Post-sanctions (2022–2025) |
|------|--------------------------|---------------------------|
| 1 | npl_ma3 | **yoy_provisions** (+12 ranks) |
| 2 | npl_ma6 | **key_rate** (+13 ranks) |
| 3 | yoy_new_loans | npl_yoy |
| 4 | coverage_lag5 | npl_mom |
| 5 | npl_ma12 | prov_rate |

Not a single variable appears in both top-5 lists. Before sanctions — inertial regime (NPL history dominates). After sanctions — macroeconomic and behavioral indicators dominate.

### 4. EWS Signal Quality (H=12)

| Metric | Value |
|--------|-------|
| ROC-AUC | **1.000** |
| Bootstrap 95% CI | **[1.000, 1.000]** |
| AUC = 1.0 in bootstrap | **97.8%** of iterations |
| Precision | 1.000 (zero false alarms) |
| Recall | 0.429 |
| F1-score | 0.600 |

Three robustness checks confirm no data leakage: train-based threshold, exclusion of lag features (R² improves: 0.87 → 0.90), bootstrap validation.

### 5. Forecast: September 2026

| Scenario | NPL Sep 2026 | Change |
|----------|--------------|--------|
| Baseline (Sep 2025 conditions) | **3.82%** | -1.6% |
| Stress (+3pp rate, +1pp unemployment, +10 RUB, -2pp GDP) | **4.15%** | +6.9% |
| Optimistic (-3pp rate, -5 RUB, +1pp GDP) | **3.78%** | -2.4% |

95% CI: **[3.53%, 4.11%]**

**Paradox:** The model forecasts NPL decline at a 18% key rate. SHAP decomposition explains this through behavioral adaptation — high prov_rate (7.71%) and yoy_provisions (+15.9%) reflect banks' preventive reserving strategy, which overrides the pressure from high rates.

### 6. Lasso vs Ridge Asymmetry

Lasso R² = 0.06 vs Ridge R² = 0.87 at H=12. This reveals that the predictive signal is **distributed** across many correlated macro variables, not concentrated in a few. L1 regularization zeros out most coefficients and loses this distributed signal. L2 regularization preserves it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              INPUT: Macroeconomic Data                  │
│  key_rate, unemployment, usd_rub, gdp_yoy, provisions   │
│  prov_rate, inflation, real_income_yoy...               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                        │
│  • Rolling stats: npl_ma3, npl_ma6, npl_ma12, npl_std   │
│  • YoY transforms: yoy_provisions, yoy_new_loans        │
│  • MoM: mom_usd_rub                                     │
│  • Lag features from CCF: coverage_lag5                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         LEVEL 1: PD-MODEL (Ridge Regression)            │
│                                                         │
│  ΔNPL_{t+H} = NPL_{t+H} - NPL_t                        │
│                                                         │
│  Walk-forward validation (expanding window)             │
│  H ∈ {3, 6, 12} months                                  │
│  α selected via RidgeCV + TimeSeriesSplit               │
│                                                         │
│  Comparison: Naive, AR(1), OLS-HAC, Lasso, Ridge        │
│  Evaluation: DM test, RMSE, R²                          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│    LEVEL 2: EWS THRESHOLD RULE                          │
│                                                         │
│  Alert = 1 if ΔNPL_predicted > Q₀.₇₅(ΔNPL_train)       │
│  (Train-based threshold — no data leakage)              │
│                                                         │
│  Metrics: ROC-AUC, Precision, Recall, F1                │
│  Bootstrap: 1000 iterations, 95% CI                     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         SHAP INTERPRETATION (LinearExplainer)           │
│                                                         │
│  • Global: feature importance ranking                   │
│  • Local: waterfall plots for key dates                 │
│  • Regime: pre- vs post-sanctions comparison            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         STRUCTURAL BREAK ANALYSIS                       │
│                                                         │
│  • Chow test (full + reduced model)                     │
│  • Interaction effects test (X × D_sanctions)           │
│  • Regime-dependent SHAP comparison                     │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

| Parameter | Value |
|-----------|-------|
| **Period** | February 2019 — September 2025 |
| **Frequency** | Monthly |
| **Raw observations** | 80 |
| **After feature engineering** | 56 |
| **Missing values** | 0 |
| **Sources** | Bank of Russia, Rosstat |

### Variables

| Group | Variable | Description | Units |
|-------|----------|-------------|-------|
| **Banking** | total_debt | Corporate loan portfolio | bln RUB |
| | overdue_debt | Overdue loans | bln RUB |
| | new_loans | New loan issuance (monthly) | bln RUB |
| | provisions | Loan loss reserves | bln RUB |
| **Risk metrics** | npl | Overdue / Total debt | ratio |
| | coverage | Provisions / Overdue debt | ratio |
| | prov_rate | Provisions / Total debt | ratio |
| | new_loans_share | New loans / Total debt | ratio |
| **Macro** | key_rate | CBR key interest rate | ratio |
| | inflation | Annual CPI | ratio |
| | unemployment | ILO methodology | ratio |
| | usd_rub | USD/RUB exchange rate | RUB |
| | gdp_yoy | Real GDP growth YoY | ratio |
| | real_income_yoy | Real disposable income growth | ratio |

**Target variable:**

```
ΔNPL_{t+H} = NPL_{t+H} − NPL_t,    H ∈ {3, 6, 12}
```

**Data limitations:**
- `gdp_yoy` and `real_income_yoy` are published quarterly by Rosstat — represented as monthly values with repetition within each quarter
- Regulatory forbearance (2022–2024) may have downward-biased official NPL figures
- Part of detailed bank-level reporting was closed by CBR after February 2022

---

## Results

### Model Comparison

| Model | H=3 RMSE | H=3 R² | H=6 RMSE | H=6 R² | H=12 RMSE | H=12 R² |
|-------|----------|--------|----------|--------|-----------|---------|
| Naive | 0.0031 | -0.96 | 0.0049 | -2.51 | 0.0065 | -0.85 |
| AR(1) | 0.0022 | -0.02 | 0.0020 | +0.39 | 0.0022 | +0.78 |
| OLS-HAC | 0.0034 | -1.44 | 0.0029 | -0.26 | 0.0010 | **+0.96** |
| Lasso | 0.0025 | -0.33 | 0.0024 | +0.14 | 0.0046 | +0.06 |
| **Ridge** | 0.0023 | -0.14 | 0.0029 | -0.28 | 0.0017 | **+0.87** |

### Diebold-Mariano Test (H=6)

|  | Naive | AR(1) | OLS-HAC | Lasso | Ridge |
|--|-------|-------|---------|-------|-------|
| **Naive** | — | +1.62 | +1.23 | +1.65* | +1.66* |
| **AR(1)** | -1.62 | — | nan | nan | -1.52 |
| **OLS-HAC** | -1.23 | nan | — | +2.44** | -0.05 |
| **Lasso** | -1.65* | nan | -2.44** | — | -1.64 |
| **Ridge** | -1.66* | +1.52 | +0.05 | +1.64 | — |

Negative DM → row model is better. `*` p<0.10, `**` p<0.05

### SHAP: Top-5 Global Feature Importance (Ridge, H=12)

```
1. prov_rate          ████████████████████ 0.00355
2. unemployment       ████████████████     0.00277
3. npl_ma3            ███████████████      0.00260
4. yoy_provisions     ████████████         0.00204
5. npl_ma12           ████████             0.00152
```

---

## Project Structure

```
NPL-Early-Warning-Model/
│
├── NPL_Early_Warning_Model.ipynb    # Main analysis notebook
├── README.md                        # This file
│
├── data/
│   └── dataset_final_clean_-_Sheet1__3_.csv   # Source dataset
│
└── outputs/
    ├── plot_npl_dynamics.png            # NPL time series with crisis markers
    ├── targets_over_time.png            # ΔNPL dynamics by horizon
    ├── horizon_effect_r2.png            # Main result: R² by horizon
    ├── shap_bar_plot.png                # Global SHAP importance
    ├── shap_beeswarm.png                # SHAP direction plot
    ├── shap_by_regimes.png              # Pre- vs post-sanctions SHAP
    ├── interaction_effects.png          # Structural break: effect changes
    ├── ews_roc_curve.png                # ROC curve (AUC=1.0)
    ├── ews_bootstrap_auc.png            # Bootstrap AUC distribution
    ├── final_forecast.png               # NPL forecast Sep 2026
    └── forecast_shap_waterfall.png      # SHAP decomposition of forecast
```

---

## Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn shap scipy
```

### Quick Start

```python
# 1. Clone repository
git clone https://github.com/dashazhrvleva/NPL-Early-Warning-Model.git
cd NPL-Early-Warning-Model

# 2. Open in Google Colab or Jupyter
# Upload: dataset_final_clean_-_Sheet1__3_.csv

# 3. Run cells sequentially
# Each section has markdown explanations before code blocks
```
