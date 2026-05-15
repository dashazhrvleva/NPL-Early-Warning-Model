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

