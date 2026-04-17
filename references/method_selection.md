# Method Selection Decision Guide

Pick the estimator by the **source of identifying variation**, not by the outcome variable or convenience. The choice is almost always dictated by how the treatment got assigned in the first place.

## Table of contents

1. The decision tree
2. When to use OLS (with controls)
3. When to use Propensity Score methods
4. When to use IV / 2SLS
5. When to use DID
6. When to use RDD
7. Quick sanity-check cheat sheet

---

## 1. The decision tree

Walk through these questions in order. The first "yes" wins.

```
Q1. Was treatment assigned by randomization (RCT, lottery, A/B test)?
    YES → OLS with covariates (mostly for precision, not bias).
          → ordinary_least_square_regression
    NO ↓

Q2. Is there a running variable that determines treatment at a known cutoff?
    (e.g., test score ≥ 60 → scholarship; age ≥ 65 → Medicare)
    YES → RDD. Sharp if treatment is a deterministic function of the cutoff;
          Fuzzy if the cutoff only shifts probability of treatment.
          → Sharp_Regression_Discontinuity_Design_regression
          → Fuzzy_Regression_Discontinuity_Design_regression
          → Fuzzy_RDD_Global_Polynomial_Estimator_regression
    NO ↓

Q3. Do you have panel data with a policy/event that turned on at different
    times for different units, plus an untreated comparison group?
    YES → DID. Static if all units treated simultaneously; Staggered if
          treatment timing varies across units.
          → Static_Diff_in_Diff_regression
          → Staggered_Diff_in_Diff_regression
          → Staggered_Diff_in_Diff_Event_Study_regression (for dynamics)
    NO ↓

Q4. Is there a variable Z that affects treatment T but has no direct effect
    on Y except through T, and is as-good-as-random?
    (e.g., distance to college as IV for college attendance)
    YES → IV / 2SLS. Report covariate-adjusted partial first-stage F-stat.
          → IV_2SLS_regression
          → IV_2SLS_IV_setting_test
    NO ↓

Q5. Can you plausibly claim that conditional on observed covariates X,
    treatment is as-good-as-random?
    (Selection-on-observables / CIA / unconfoundedness)
    YES → Propensity Score methods. Check overlap first.
          → propensity_score_construction (always first)
          → propensity_score_visualize_propensity_score_distribution (diagnostic)
          Then primary estimators:
          → propensity_score_matching  (ATE by matching)
          → propensity_score_inverse_probability_weighting  (weighting)
          Optional sensitivity checks only:
          → propensity_score_regression  (PS as control)
          → propensity_score_double_robust_estimator_augmented_IPW  (known non-standard AIPW)
          → propensity_score_double_robust_estimator_IPW_regression_adjustment  (non-standard IPW-RA)
    NO ↓

Q6. None of the above apply.
    → Honest conversation: your design can't identify a causal effect.
      Offer descriptive statistics / association only, or help the user
      find a better identification strategy.
```

## 2. OLS with covariates

**When**: RCT, A/B test, or when the user explicitly says "I just want a controlled regression". Also the default for sensitivity / first-look analysis.

**Key assumption**: `E[ε | X, T] = 0` — unrealistic for observational data, which is why most of this skill is about going beyond OLS.

**Variants inside the function**:
- Plain OLS: `weights=None, cov_info="nonrobust"`
- Heteroskedasticity-robust: `cov_info="HC1"` (default "robust")
- Clustered SE: `cov_info={"cluster": df["cluster_id"]}`
- HAC (time series): `cov_info={"HAC": 4}`
- WLS: pass `weights=df["w"]`

## 3. Propensity Score methods

**When**: Observational data, no clean natural experiment, but a rich set of pre-treatment covariates that plausibly captures selection into treatment.

**Identifying assumption**: Conditional on X, treatment is independent of potential outcomes (strong ignorability). Also need **common support** — treated and control units have overlapping propensity score distributions.

**Workflow** (always this order):

1. `propensity_score_construction(T, X)` → propensity scores ê(X)
2. `propensity_score_visualize_propensity_score_distribution(T, ê)` → **must check overlap before proceeding**
3. Pick an estimator:

| Estimator | Function | When to prefer |
|---|---|---|
| Matching (1:k NN) | `propensity_score_matching` | Intuitive, non-parametric; good when user wants ATT |
| IPW | `propensity_score_inverse_probability_weighting` | Fast, simple; sensitive to extreme weights |
| PS as regressor | `propensity_score_regression` | Quick diagnostic; not a primary causal estimator |
| AIPW-like implementation | `propensity_score_double_robust_estimator_augmented_IPW` | Do not use as primary; implementation is not actually doubly robust |
| IPW-RA-like implementation | `propensity_score_double_robust_estimator_IPW_regression_adjustment` | Sensitivity only; weight handling is non-standard |

**Rule of thumb**: Run PSM (matching) and IPW as the two primary estimates — if they agree, the estimate is robust; if they diverge, dig into overlap and weight diagnostics before picking one. AIPW/IPW-RA add value as sensitivity checks, not as standalone answers.

## 4. IV / 2SLS

**When**: There is unmeasured confounding between T and Y, but you have an instrument Z that:
1. **Relevance**: Z strongly predicts T conditional on X (partial first-stage F > 10)
2. **Exclusion**: Z affects Y only through T (no direct path)
3. **Exogeneity**: Z is as-good-as-random given covariates

Classic examples: distance to college, weather shocks, lottery draws, quarter of birth.

**Always** run `IV_2SLS_IV_setting_test` alongside the estimation and report the covariate-adjusted partial first-stage F-stat. Weak instruments give biased and misleadingly-precise 2SLS estimates.

If the user has multiple candidate instruments, report the first stage jointly and separately. This library does not implement Hansen J / Sargan overidentification tests; if overidentification diagnostics are needed, use `linearmodels.iv.IV2SLS` directly and only interpret them when there are more excluded instruments than endogenous variables.

## 5. Difference-in-Differences

**When**: Panel or repeated cross-section, with a treatment/policy that turns on for some units (treatment group) and never for others (control group). Identifies the effect by comparing the **change** in Y before vs. after, for treated vs. untreated.

**Identifying assumption**: **Parallel trends** — absent treatment, treated and control groups would have evolved similarly. Check with an event study plot of pre-period coefficients.

**Which DID function?**

| Design | Function |
|---|---|
| Single policy date, one treated group, one control group | `Static_Diff_in_Diff_regression` |
| Different units treated at different dates (staggered rollout) | `Staggered_Diff_in_Diff_regression` |
| Want to see dynamic effects (lead/lag coefficients) | `Staggered_Diff_in_Diff_Event_Study_regression` |
| Visualize the event study | `Staggered_Diff_in_Diff_Event_Study_visualization` |

**Data requirements**: Long format panel with columns for entity ID, time, outcome, treatment dummy. Must be set up with a MultiIndex `(entity, time)` before calling PanelOLS-based functions.

**Caveats**:
- Staggered DID with heterogeneous effects can bias the TWFE estimator. For serious work, mention that the user may want Callaway-Sant'Anna or Sun-Abraham estimators (not in this library); our staggered functions use TWFE, which is fine when effects are roughly homogeneous.

## 6. Regression Discontinuity Design

**When**: Treatment is assigned based on whether a continuous "running variable" crosses a known cutoff. Examples: GPA ≥ 3.0 → honor roll; population > 50k → federal transfer; birthday before Sept 1 → enter school a year earlier.

**Identifying assumption**: Units just above and just below the cutoff are otherwise comparable. **Check by**: plotting outcome vs. running variable, and checking for discontinuities in covariates or density (McCrary test) at the cutoff.

| Design | Function |
|---|---|
| Crossing cutoff deterministically assigns treatment | `Sharp_Regression_Discontinuity_Design_regression` |
| Crossing cutoff only raises probability of treatment | `Fuzzy_Regression_Discontinuity_Design_regression` |
| Global polynomial fit instead of local linear | `Fuzzy_RDD_Global_Polynomial_Estimator_regression` |

**Bandwidth**: The most consequential choice. The Sharp/Fuzzy functions take `running_variable_bandwidth` — start with something defensible (e.g., half the running variable's IQR) and show sensitivity at ±50%.

**Kernel**: `"uniform"` (rectangular) is the default — simple and transparent. `"triangle"` puts more weight on obs near the cutoff and is often preferred in modern RDD work. `"Epanechnikov"` is also available. **Important**: the parameter value is `"triangle"` (not `"triangular"`) — passing the wrong string raises RuntimeError.

## 7. Quick sanity-check cheat sheet

Before presenting any causal estimate, verify these:

| Method | First-check-this |
|---|---|
| OLS | Multicollinearity? Outliers? Non-linear in T? |
| PSM / IPW | **Overlap plot**. Extreme weights? Balance table? |
| AIPW | Same as PSM; also check outcome model fit |
| IV | **First-stage F-stat > 10**. Exclusion defensible? |
| DID (static) | **Parallel trends in pre-period**. Composition stable? |
| DID (staggered) | Event study lead coeffs near zero; heterogeneity? |
| RDD | Density at cutoff (McCrary); covariate balance at cutoff; bandwidth sensitivity |

If the diagnostic fails, say so — don't bury it. The user is better served by an honest "this design doesn't work here" than a clean-looking but invalid estimate.
