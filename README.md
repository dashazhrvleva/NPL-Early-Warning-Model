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
| Chow test (top-5 SHAP variables) | F = 5.22 | < 0.001 | Significant at 0.1% level |
| Interaction test | 6 variables | p < 0.10 | 4 sign changes |

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
# Upload: dataset.csv

# 3. Run cells sequentially
# Each section has markdown explanations before code blocks
```
### Notebook Sections

```
Part 0:   Setup & Imports
Part I:   Exploratory Data Analysis
  1.1     NPL dynamics + crisis markers
  1.2     Descriptive statistics (skew, kurtosis)
  1.3     Time series visualization
  1.4     Stationarity tests (ADF + KPSS)
  1.5     Correlation analysis (Pearson + Spearman)
  1.6     Cross-correlation functions (CCF)
  1.7     Granger causality
  1.8     Multicollinearity (VIF)
  1.9     Feature engineering + final dataset

Part II:  Model Comparison
  2.1     Walk-forward validation setup
  2.2     Baselines: Naive + AR(1)
  2.3     OLS with HAC standard errors
  2.4     Lasso (L1 regularization)
  2.5     Ridge (L2 regularization)
  2.6     Diebold-Mariano test

Part III: SHAP Interpretation
  3.1     Global feature importance (bar plot)
  3.2     Effect distribution (beeswarm)
  3.3     Local explanation (waterfall)
  3.4     Temporal SHAP evolution

Part IV:  Structural Break Analysis
  4.1     Chow test (full + reduced model)
  4.2     Interaction effects test
  4.3     Regime-dependent SHAP

Part V:   EWS Signal Evaluation
  5.1     Threshold rule + classification metrics
  5.2     ROC curve (AUC)
  5.3     Robustness: without lag features
  5.4     Bootstrap validation (1000 iterations)

Part VI:  Forecast 2026
  6.1     Final model training
  6.2     Point forecast + 95% CI
  6.3     SHAP decomposition of forecast
  6.4     Scenario analysis
```

---

## Technical Notes

### Walk-Forward Validation

No future data is used in training at any point:

```
Fold 1: Train[1..40]  → Predict[41]
Fold 2: Train[1..41]  → Predict[42]
...
Fold n: Train[1..n]   → Predict[n+1]
```

Standardization (`StandardScaler`) is applied within each fold on training data only. Regularization parameters (α) are selected via `TimeSeriesSplit` within the training window.

### No Data Leakage — Three Checks

1. **Threshold:** computed on training data within each fold
2. **Lag features:** removing them *improves* R² (0.87 → 0.90), confirming no look-ahead bias
3. **Bootstrap AUC:** 95% CI = [1.000, 1.000] across 1,000 iterations

### Why Ridge over Lasso

At H=12, Lasso achieves R²=0.06 while Ridge achieves R²=0.87. This 14x gap is explained by the **distributed signal hypothesis**: the NPL forecast signal is spread across many correlated macro variables. L1 regularization zeros out most of them, losing the signal. L2 shrinks without zeroing, preserving it.

---

## References

1. Acharya, V.V. (2009). A theory of systemic risk and design of prudential bank regulation. *Journal of Financial Stability*, 5(3), 224–255.

2. Drehmann, M., & Juselius, M. (2014). Evaluating early warning indicators of banking crises: Satisfying policy requirements. *International Journal of Forecasting*, 30(3), 759–780.

3. Lundberg, S., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4768–4777.

4. Diebold, F.X., & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

5. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

6. Bussière, M., & Fratzscher, M. (2006). Towards a new early warning system of financial crises. *Journal of International Money and Finance*, 25(6), 953–973.

7. FSA Japan (2025). Attempt to identify early warning signals on credit risks using regional banks' loan data and macroeconomic indicators. *FSA Analytical Notes*, vol. 3.

---

## About

**Research context:** Master's thesis, HSE University, 2026
**Data sources:** Bank of Russia (banking statistics), Rosstat (macroeconomic indicators)

---

<div align="center">
<sub>Built with Python · pandas · statsmodels · scikit-learn · SHAP</sub>
</div>
