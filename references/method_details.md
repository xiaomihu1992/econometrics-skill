# Method Details — exact signatures and minimal snippets

All functions live in `lib/econometric_algorithm.py`. Return values vary — read the "Returns" line carefully before using.

Convention: every function accepts `target_type` to control what's returned. In day-to-day analysis use `target_type="final_model"` (or `"estimator"` for Fuzzy RDD); the other values (`"neg_pvalue"`, `"rsquared"`) exist for automated hyperparameter search and usually aren't what you want.

## Table of contents

1. [OLS](#1-ols)
2. [Propensity score — construction & viz](#2-propensity-score--construction--viz)
3. [PS matching (PSM)](#3-ps-matching-psm)
4. [Inverse probability weighting (IPW)](#4-ipw)
5. [PS regression](#5-ps-regression)
6. [AIPW-like estimator (not doubly robust)](#6-aipw-like-estimator-not-doubly-robust)
7. [IPW-RA-like estimator (non-standard weights)](#7-ipw-ra-like-estimator-non-standard-weights)
8. [IV / 2SLS](#8-iv--2sls)
9. [IV setting test](#9-iv-setting-test)
10. [Static DID](#10-static-did)
11. [Staggered DID](#11-staggered-did)
12. [DID Event Study](#12-did-event-study)
13. [DID Event Study visualization](#13-did-event-study-visualization)
14. [Sharp RDD](#14-sharp-rdd)
15. [Fuzzy RDD (two-step)](#15-fuzzy-rdd-two-step)
16. [Fuzzy RDD global polynomial](#16-fuzzy-rdd-global-polynomial)
17. [Utilities: load_table & get_column_info](#17-utilities-load_table--get_column_info)

---

## 1. OLS

```python
ordinary_least_square_regression(
    dependent_variable,      # pd.Series
    treatment_variable,      # pd.Series
    covariate_variables,     # pd.DataFrame or None
    weights=None,            # pd.Series or None (WLS if provided)
    cov_info="nonrobust",    # "nonrobust" | "HC0..HC3" | {"HAC": int} | {"cluster": pd.Series}
    target_type="final_model",
    output_tables=False,
) -> statsmodels RegressionResults
```

Minimal use:
```python
m = ordinary_least_square_regression(df["Y"], df["T"], df[["X1","X2"]], cov_info="HC1")
print(m.summary())
ate = m.params["T"]
```

---

## 2. Propensity score — construction & viz

```python
propensity_score_construction(treatment_variable, covariate_variables) -> pd.Series  # named "propensity_score"

propensity_score_visualize_propensity_score_distribution(treatment_variable, propensity_score) -> matplotlib.figure.Figure
```

**Logit output suppression**: `propensity_score_construction` calls `sm.Logit(...).fit()` without `disp=0`, so it prints convergence info to stdout on every call. Inside a bootstrap loop (500+ iterations), this produces massive output. Suppress with `contextlib.redirect_stdout(io.StringIO())` around the call (see `examples/psm_example.py` for the pattern).

Always run both before any PSM / IPW / AIPW analysis:
```python
ps = propensity_score_construction(df["T"], df[["X1","X2","X3"]])
fig = propensity_score_visualize_propensity_score_distribution(df["T"], ps)
fig.savefig("overlap.png", dpi=150, bbox_inches="tight")
```

**Interpretation**: If the treated and control histograms don't overlap, stop — PS methods won't work. Discuss trimming (drop units with ê < 0.05 or > 0.95) or a different identification strategy.

---

## 3. PS Matching (PSM)

```python
propensity_score_matching(
    dependent_variable,
    treatment_variable,
    propensity_score,       # output of propensity_score_construction
    matched_num=1,          # 1:k nearest-neighbor matching
    target_type="ATE",      # "ATE" | "ATT" (ATC not supported)
) -> float  # point estimate only — no SE or CI; use bootstrap for inference
```

```python
ate = propensity_score_matching(df["Y"], df["T"], ps, matched_num=1, target_type="ATE")
```

**No standard error is returned.** For inference, bootstrap the entire pipeline (PS estimation + matching):

```python
import numpy as np
rng = np.random.default_rng(42)
n = len(df)
boot_ates = []
for _ in range(500):
    idx = rng.choice(n, size=n, replace=True)
    boot_df = df.iloc[idx].reset_index(drop=True)
    boot_ps = propensity_score_construction(boot_df["T"], boot_df[["X1","X2"]])
    boot_ate = propensity_score_matching(boot_df["Y"], boot_df["T"], boot_ps, target_type="ATE")
    boot_ates.append(boot_ate)
se = np.std(boot_ates)
ci_low, ci_high = np.percentile(boot_ates, [2.5, 97.5])
print(f"ATE = {ate:.3f}, SE = {se:.3f}, 95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
```

This same bootstrap pattern works for IPW — just swap `propensity_score_matching` with `propensity_score_inverse_probability_weighting` inside the loop.

---

## 4. IPW

```python
propensity_score_inverse_probability_weighting(
    dependent_variable,
    treatment_variable,
    propensity_score,
    target_type="ATE",      # "ATE" | "ATT" (ATC not supported)
) -> float  # point estimate only — no SE or CI; use bootstrap for inference
```

```python
ate = propensity_score_inverse_probability_weighting(df["Y"], df["T"], ps, target_type="ATE")
```

**Watch for**: extreme weights. If any ê is very close to 0 or 1, that unit gets a huge weight and dominates. Check with `ps.describe()` before reporting.

---

## 5. PS regression

```python
propensity_score_regression(
    dependent_variable,
    treatment_variable,
    propensity_score,
    cov_type=None,          # None or "nonrobust" / "HC1" / ...
    target_type="final_model",
    output_tables=False,
) -> statsmodels RegressionResults
```

Uses the PS as a single control. Quick but less defensible than AIPW.

---

## 6. AIPW-like estimator (not doubly robust)

This function is kept for compatibility with the original project, but it should not be treated as a primary doubly robust estimator.

```python
propensity_score_double_robust_estimator_augmented_IPW(
    dependent_variable,
    treatment_variable,
    propensity_score,
    covariate_variables,    # pd.DataFrame, used to fit outcome model
    cov_type=None,
) -> float  # ATE
```

```python
ate = propensity_score_double_robust_estimator_augmented_IPW(
    df["Y"], df["T"], ps, df[["X1","X2","X3"]], cov_type="HC1",
)
```

**Implementation note — this estimator is NOT doubly robust**:

1. Uses a single OLS of Y on T + X (rather than separate outcome models for treated and control groups).
2. Only constructs counterfactuals in one direction (predicts Y(1) for controls, but never predicts Y(0) for treated).
3. The canonical AIPW (Robins-Rotnitzky-Zhao 1994) requires both E[Y(1)|X] and E[Y(0)|X] for all units — this implementation is asymmetric and loses the double-robustness property.

In smoke testing, this function returned ~0 on a DGP where the true ATE was 3.0 (while PSM and IPW correctly estimated ~3.0). **Do not use as a standalone estimator.** Use only as one data point among multiple estimates, and weight PSM/IPW results more heavily. For a proper doubly-robust estimator, use `econml.dr.DRLearner` or `dowhy`.

---

## 7. IPW-RA-like estimator (non-standard weights)

```python
propensity_score_double_robust_estimator_IPW_regression_adjustment(
    dependent_variable,
    treatment_variable,
    covariate_variables,
    propensity_score,
    cov_type=None,
    target_type="final_model",
    output_tables=False,
) -> statsmodels RegressionResults  # WLS-based
```

Compatibility estimator: weight by 1/ê (for treated) or 1/(1-ê) (for controls), then run WLS with covariates. Use only as a sensitivity check because the implementation changes the weights before WLS.

**Watch the argument order**: this is the only PS function where `covariate_variables` comes *before* `propensity_score`. All other PS functions put `propensity_score` first. Getting this wrong will silently produce garbage results.

**Implementation note**: the library applies `sqrt` to the IPW weights before running WLS (line 292: `IPW = IPW ** 0.5`). This is non-standard and not justified in the DR literature. Interpret with caution and cross-check against PSM/IPW.

---

## 8. IV / 2SLS

```python
IV_2SLS_regression(
    dependent_variable,
    treatment_variable,
    IV_variable,             # pd.Series or pd.DataFrame (for multiple instruments)
    covariate_variables,     # pd.DataFrame or None
    cov_info="nonrobust",    # same options as OLS
    target_type="final_model",
    output_tables=False,
) -> statsmodels OLS RegressionResults  # NOT linearmodels IV2SLS — this is a hand-rolled 2SLS
```

```python
m = IV_2SLS_regression(df["Y"], df["T"], df["Z"], df[["X1","X2"]], cov_info="HC1")
```

**Always pair with** `IV_2SLS_IV_setting_test` to report the first-stage F.

**CAVEAT**: This is a hand-rolled 2SLS (two sequential OLS calls), NOT `linearmodels.iv.IV2SLS`. The second-stage standard errors are not adjusted for first-stage estimation uncertainty, so **reported SEs and p-values will be too small**. For publication-quality inference, cross-check with `linearmodels.iv.IV2SLS` or R's `ivreg`.

---

## 9. IV setting test

```python
IV_2SLS_IV_setting_test(
    dependent_variable,
    treatment_variable,
    IV_variable,
    covariate_variables,
    cov_type=None,
) -> dict  # keys: "relevant_condition", "exclusion_restriction"
```

Returns a dict containing **two statsmodels RegressionResults objects** (not raw numbers):
- `"relevant_condition"` — first-stage OLS of T on Z. Extract the F-stat via `.fvalue` and the p-value via `.f_pvalue`.
- `"exclusion_restriction"` — OLS of the second-stage residuals on Z. A significant coefficient suggests Z violates the exclusion restriction.

```python
diag = IV_2SLS_IV_setting_test(df["Y"], df["T"], df["Z"], df[["X1","X2"]])
first_stage = diag["relevant_condition"]
print(f"First-stage F = {first_stage.fvalue:.2f} (p = {first_stage.f_pvalue:.4f})")
print(f"First-stage R² = {first_stage.rsquared:.4f}")
print(first_stage.summary())

excl = diag["exclusion_restriction"]
print(f"Exclusion test: coeff on Z = {excl.params[df['Z'].name]:.4f}, p = {excl.pvalues[df['Z'].name]:.4f}")
```

Rule of thumb: first-stage F > 10 for a strong instrument. Much less and the 2SLS estimate is unreliable (biased toward OLS).

---

## 10. Static DID

**Data requirement**: `df` must have a MultiIndex `(entity_id, time)`.

```python
Static_Diff_in_Diff_regression(
    dependent_variable,           # pd.Series with (entity, time) index
    treatment_entity_dummy,       # 1 if unit is in treated group (ever)
    treatment_finished_dummy,     # 1 if time >= treatment start (for everyone)
    covariate_variables,
    entity_effect=False,
    time_effect=False,
    other_effect=None,
    cov_type="unadjusted",        # "unadjusted" | "robust" | "cluster_entity" | "cluster_time" | "cluster_both"
    target_type="final_model",
    output_tables=False,
) -> linearmodels PanelOLS result
```

The coefficient of interest is named `"treatment_group_treated"` (the interaction term, i.e. the DID estimator).

```python
df = df.set_index(["firm_id","year"])
m = Static_Diff_in_Diff_regression(
    df["Y"], df["treated_firm"], df["post_policy"], df[["X1"]],
    entity_effect=True, time_effect=True, cov_type="cluster_entity",
)
ate = m.params["treatment_group_treated"]
```

---

## 11. Staggered DID

Same index requirement.

```python
Staggered_Diff_in_Diff_regression(
    dependent_variable,
    entity_treatment_dummy,       # 1 if unit is currently treated at this (entity, time)
    covariate_variables,
    entity_effect=True,           # defaults to TRUE here (two-way FE)
    time_effect=True,
    other_effect=None,
    cov_type="unadjusted",
    target_type="final_model",
    output_tables=False,
) -> linearmodels PanelOLS result
```

Coefficient of interest: `"treatment_entity_treated"`. Note the TWFE caveat (see method_selection.md §5).

---

## 12. DID Event Study

```python
Staggered_Diff_in_Diff_Event_Study_regression(
    dependent_variable,
    entity_treatment_dummy,
    covariate_variables,
    see_back_length=4,      # number of pre-periods (leads); MINIMUM 4
    see_forward_length=3,   # number of post-periods (lags); MINIMUM 3
    entity_effect=True,
    time_effect=True,
    other_effect=None,
    cov_type="unadjusted",
    target_type="final_model",
    output_tables=False,
) -> linearmodels PanelOLS result
```

**Hard constraints**: `see_back_length >= 4` and `see_forward_length >= 3` are enforced; smaller values raise RuntimeError. Also requires `see_back_length + see_forward_length < number_of_time_periods`.

**Coefficient naming convention** (with default `see_back_length=4, see_forward_length=3`):

| Coefficient name | Meaning |
|---|---|
| `Lead_D4+` | 4+ periods before treatment (binned) |
| `Lead_D3` | 3 periods before treatment |
| `Lead_D2` | 2 periods before treatment |
| *(Lead_D1 omitted)* | *baseline / reference period (t = -1)* |
| `D0` | Treatment onset period |
| `Lag_D1` | 1 period after treatment |
| `Lag_D2` | 2 periods after treatment |
| `Lag_D3+` | 3+ periods after treatment (binned) |

The omitted reference period is always `Lead_D1` (one period before treatment). All lead/lag coefficients are relative to this baseline.

To extract a specific coefficient:
```python
m = Staggered_Diff_in_Diff_Event_Study_regression(...)
immediate_effect = m.params["D0"]
pre_trend_check = m.params["Lead_D3"]  # should be near zero
```

Produces lead/lag coefficients. Use the visualization below to plot them.

---

## 13. DID Event Study visualization

```python
Staggered_Diff_in_Diff_Event_Study_visualization(
    regression_model,       # result from Event_Study_regression above
    see_back_length=4,
    see_forward_length=3,
) -> matplotlib.figure.Figure
```

**Parallel-trends check**: the leads (k < 0) should bounce around zero. If they trend, the design is suspect.

---

## 14. Sharp RDD

```python
Sharp_Regression_Discontinuity_Design_regression(
    dependent_variable,
    entity_treatment_dummy,        # 1 iff running_variable >= cutoff
    running_variable,
    covariate_variables,
    running_variable_cutoff,       # float
    running_variable_bandwidth,    # float or None (None = full sample)
    kernel_choice="uniform",       # "uniform" | "triangle" | "Epanechnikov"
    cov_info="nonrobust",
    target_type="final_model",
    output_tables=False,
) -> statsmodels RegressionResults (WLS)
```

Local linear regression with kernel weighting. The coefficient on `entity_treatment_dummy` is the RDD ATE at the cutoff.

```python
treatment = (df["entry_score"] >= 60).astype(int)
treatment.name = "scholarship"  # the coefficient will use this name
m = Sharp_Regression_Discontinuity_Design_regression(
    df["test_score_after"], treatment, df["entry_score"],
    covariate_variables=None,
    running_variable_cutoff=60, running_variable_bandwidth=10,
    kernel_choice="triangle", cov_info="HC1",
)
ate = m.params["scholarship"]  # uses the Series .name attribute
```

**Important**: The treatment coefficient in the regression uses the `.name` attribute of the `entity_treatment_dummy` Series you pass in. Make sure to set a clean name before calling the function — names with spaces or special characters will break the formula API.

**Always show sensitivity** to bandwidth (half and double the chosen value).

---

## 15. Fuzzy RDD (two-step)

```python
Fuzzy_Regression_Discontinuity_Design_regression(
    dependent_variable,
    entity_treatment_dummy,           # actual treatment, not assignment
    running_variable,
    covariate_variables,
    running_variable_cutoff,
    running_variable_bandwidth,
    kernel_choice="uniform",
    cov_info="nonrobust",
    target_type="estimator",          # "estimator" (returns float ATE) or "final_models" (returns list)
    output_tables=False,
)
```

Two-step fuzzy RDD: reduced form / first stage. Assignment indicator is implicit from the running variable and cutoff.

Return types:
- `target_type="estimator"` → `float` (the Wald/fuzzy-RDD ATE estimate)
- `target_type="final_models"` → `list` of two OLS model objects: `[reduced_form_model, first_stage_model]`

Note: unlike Sharp/Global-Polynomial RDD, Fuzzy RDD does not take `running_variable_bandwidth=None` as "use full sample" — bandwidth is always required and must be positive.

---

## 16. Fuzzy RDD global polynomial

```python
Fuzzy_RDD_Global_Polynomial_Estimator_regression(
    dependent_variable,
    entity_treatment_dummy,
    running_variable,
    covariate_variables,
    running_variable_cutoff,
    max_order,                       # int, polynomial order cap
    kernel_choice="uniform",
    cov_info="nonrobust",
    target_type="final_model",
    output_tables=False,
) -> statsmodels RegressionResults
```

Fits polynomials up to `max_order` on both sides of the cutoff. Modern RDD practice tends to prefer local linear with tuned bandwidth over high-order global polynomials (Gelman & Imbens 2019), but this function is useful for robustness checks.

---

## 17. Utilities: load_table & get_column_info

```python
from data_preprocess import load_table, get_column_info

# Load tabular data from CSV, TSV, or Excel files
load_table(
    file_path,          # str or Path — .csv, .tsv, .xlsx, .xls, .xlsm
    sheet_name=0,       # Excel sheet name or index; ignored for CSV/TSV
    **kwargs,           # extra args passed to pd.read_csv / pd.read_excel
) -> pd.DataFrame

# Categorize DataFrame columns by type
get_column_info(df) -> dict  # {"Category": [...], "Numeric": [...], "Datetime": [...], "Others": [...]}
```

Always run both after loading a dataset: `load_table` to read the file, then `get_column_info` to confirm treatment/outcome/covariate column types with the user before fitting anything.
