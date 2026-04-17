---
name: econometrics
description: "Causal inference and applied econometric analysis on tabular data, from uploaded-data diagnostics and cleaning advice to treatment-effect estimation and publication-grade applied research design. Use for policy impact, ATE/ATT/LATE/ITT, OLS, propensity scores, IV/2SLS, DID/event studies, RDD, robustness checks, falsification tests, identification memos, 因果推断, 政策评估, 数据清洗建议, 稳健性检验, and 异质性分析."
metadata: {"short-description": "Causal inference for tabular data", "version": "1.3.0", "author": "econometrics-agent"}
---

# Econometrics Skill

This skill gives Codex a curated library of **17 causal-inference estimators** (in `lib/econometric_algorithm.py`) plus the judgment to pick the right one for the user's identification strategy. For advanced research projects, referee-response, or publication-grade work, it also provides a structured applied-econometrics workflow.

## What this skill is for

Causal inference on tabular data — answering **"what is the effect of treatment T on outcome Y, holding confounders fixed?"** using methods that are defensible in applied economics / social science work.

This is **not** for pure prediction, forecasting, time-series ARIMA, or ML model training. If the user wants prediction accuracy rather than an unbiased causal estimate, redirect — don't force these tools.

## Runtime requirements

Use Python 3.10+ with `numpy`, `pandas`, `matplotlib`, `statsmodels`, `linearmodels`, `scipy`, `openpyxl` for `.xlsx/.xlsm` Excel files, and `xlrd` for legacy `.xls` Excel files.

## Depth modes

- **Quick mode**: Use the core workflow below when the user needs a defensible first-pass estimate, exploratory causal analysis, or method selection.
- **Advanced applied mode**: Use `references/advanced_applied_workflow.md` when the user asks for publication-grade analysis, referee-grade robustness, identification critique, heterogeneity, falsification tests, or a research design memo.
- **Project workflow mode**: Use `references/applied_project_workflow.md` when the user needs an end-to-end empirical project plan from question to final report.
- **Data diagnostic mode**: Use `lib/data_preprocess.py` and `references/data_preprocessing_advice.md` when the user uploads data or asks what cleaning is needed.
- **Table mode**: Use `lib/result_tables.py` and `references/result_tables.md` when the user needs compact model comparison tables.
- **Checklist mode**: Use `references/diagnostic_checklist.md` before presenting estimates as causal.

## The core workflow

Every analysis follows the same three-step shape. Don't skip steps — skipping identification makes the numbers meaningless.

### Step 1: Understand the identification strategy

Before touching any code, resolve these four questions with the user (ask if they're unstated):

1. **What is the outcome Y?** (continuous / binary / count)
2. **What is the treatment T?** (binary / continuous / policy dummy)
3. **Why would a naive `Y ~ T` regression be biased?** (selection? reverse causality? omitted variables?)
4. **What's the source of identifying variation?** (randomization / conditional independence / instrument / policy timing / cutoff)

The answer to #4 picks the method family. See `references/method_selection.md` for the full decision guide — read it whenever the identification strategy isn't obvious.

**Quick map:**

| Identifying variation | Method family | Functions |
|---|---|---|
| Randomized / as-if random | OLS with controls | `ordinary_least_square_regression` |
| Selection on observables (rich covariates) | PS methods | `propensity_score_construction`, overlap visualization, PSM/IPW primary estimates |
| Exogenous instrument Z affecting Y only via T | IV / 2SLS | `IV_2SLS_regression`, `IV_2SLS_IV_setting_test` |
| Policy change + panel data (pre/post × treated/control) | DID | `Static_Diff_in_Diff_regression`, `Staggered_Diff_in_Diff_*` (3 funcs) |
| Sharp threshold in a running variable | RDD | `Sharp_*`, `Fuzzy_*` (3 funcs) |

### Step 2: Inspect the data, then call the algorithm

Load the dataset with `load_table()` from `lib.data_preprocess`, then run `analyze_dataset()` and `format_dataset_report()` when the user uploads data or asks for cleaning guidance. Use the diagnostic report to identify likely roles, missingness, duplicates, type conversion needs, outliers, and panel structure before choosing an estimator. See `references/data_preprocessing_advice.md` for the full workflow.

After the data diagnostic pass, confirm column names with the user, then **call the algorithm directly** — these are plain Python functions returning fitted models, not agents.

Data is passed as `pd.Series` / `pd.DataFrame`:
- `dependent_variable` = `df["Y"]`
- `treatment_variable` = `df["T"]`
- `covariate_variables` = `df[["X1", "X2", ...]]` (DataFrame, or `None`)

Return values vary by function — fitted model objects (statsmodels / linearmodels), scalar ATEs, pd.Series (propensity scores), dict summaries, or matplotlib Figures. Invalid `target_type` values raise `ValueError`; when unsure, check `references/method_details.md` before calling.

See `references/method_details.md` for exact signatures, parameter semantics, and minimal code snippets for each of the 17 functions. Read it when you're about to invoke a method you haven't used before in this session.

### Step 3: Interpret in plain language

Don't dump `summary()` output and call it done. Translate:
- **Point estimate**: what magnitude and sign, in the outcome's units
- **Statistical significance**: p-value vs. conventional thresholds, but don't fetishize p<0.05
- **Practical significance**: is the effect size meaningful in the domain?
- **Caveats**: which identification assumption is most fragile here?

See `references/interpretation.md` for templates per method family (parallel trends for DID, exclusion restriction for IV, common support for PSM, etc.).

### Advanced mode deliverables

For serious applied work, do more than estimate a coefficient. Produce these artifacts:

1. **Estimand statement**: ATE/ATT/LATE/ITT, target population, time horizon, and outcome scale.
2. **Identification memo**: identifying variation, assumptions, why naive comparisons fail, and what would falsify the design.
3. **Specification grid**: baseline, preferred, richer controls, fixed effects, clustering choice, sample restrictions, and alternative functional forms.
4. **Diagnostics**: overlap/balance for PS, first-stage/reduced-form for IV, pre-trends for DID, density/covariate continuity for RDD.
5. **Robustness suite**: placebo outcomes, placebo treatment timing/cutoffs, bandwidth or trimming sensitivity, cluster choices, and influential-observation checks.
6. **Heterogeneity plan**: pre-specified subgroups or interactions, multiple-testing caution, and whether effects should be interpreted as exploratory.
7. **Research-grade caveats**: which assumption remains untestable, what evidence supports it, and what design would be stronger.

For full project scaffolding, read `references/applied_project_workflow.md`. For method-specific diagnostics, read `references/diagnostic_checklist.md`. For model comparison tables, use `lib/result_tables.py` with `references/result_tables.md`.

## Data preprocessing (before calling any algorithm)

All library functions expect clean numeric data. For uploaded datasets, first run:

```python
from data_preprocess import analyze_dataset, format_dataset_report

analysis = analyze_dataset("data.xlsx", sheet_name=0)
print(format_dataset_report(analysis))
```

Then do these checks after loading and before calling any estimator — skipping them is the #1 source of cryptic errors:

1. **Missing values**: Drop or impute NaNs before passing data. The library functions call `.astype(float)` which turns NaN-containing object columns into errors. Check with `df.isnull().sum()` and handle explicitly.

2. **Column name hygiene**: RDD functions use the statsmodels formula API (`smf.wls(formula, ...)`), which breaks on column names containing spaces, parentheses, dashes, or other special characters. Rename columns to clean identifiers before use: `df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)`.

3. **Treatment variable validation**: Most functions require a binary (0/1) treatment dummy. Verify with `df["T"].unique()` — values like 1/2, True/False, or "treated"/"control" will cause errors or silent misinterpretation. Cast explicitly: `df["T"] = (df["T"] == "treated").astype(int)`.

4. **Series `.name` attribute**: RDD and OLS functions use `series.name` to label coefficients in the output. If the Series has no name (e.g., from a computed column), set it explicitly: `treatment.name = "scholarship"`.

5. **Panel data MultiIndex for DID**: All DID functions require a `(entity, time)` MultiIndex. Set it before calling: `df = df.set_index(["firm_id", "year"])`. The entity and time levels must be sortable.

6. **Categorical encoding**: Category/string columns must be dummy-encoded before use as covariates. Use `pd.get_dummies(df[["industry"]], drop_first=True)` and concatenate with numeric covariates.

## Running the code

The algorithms live at `<skill_dir>/lib/econometric_algorithm.py`. From a working directory, add the skill's `lib/` to `sys.path` or import via the full path. A reusable pattern:

```python
import sys, os
SKILL_LIB = os.path.join(os.path.dirname(__file__), "lib")  # adjust to skill path
sys.path.insert(0, SKILL_LIB)

import pandas as pd
from econometric_algorithm import (
    ordinary_least_square_regression,
    propensity_score_construction,
    propensity_score_inverse_probability_weighting,
    IV_2SLS_regression,
    Static_Diff_in_Diff_regression,
    Sharp_Regression_Discontinuity_Design_regression,
)
from data_preprocess import analyze_dataset, format_dataset_report, get_column_info, load_table

df = load_table("data.xlsx", sheet_name=0)  # also supports .csv, .tsv, .xls, .xlsm
print(get_column_info(df))
analysis = analyze_dataset(df)
print(format_dataset_report(analysis))
```

When running the code:
- Always `matplotlib.use("Agg")` is already set inside the module — figures don't need a display
- Save figures with `fig.savefig("out.png", dpi=150, bbox_inches="tight")` rather than trying to show them
- For panel methods (DID), the DataFrame must have a **MultiIndex of (entity, time)** — see `method_details.md` DID section

## Covariance / standard errors

**The OLS/RDD/IV family and the DID family use different parameter spaces.** Mixing them up causes RuntimeError. Check which library a function uses before passing `cov_type`.

### For OLS, RDD, IV, PS-regression, IPW-RA (statsmodels-based)

Parameter name: `cov_info` (or `cov_type` for some PS functions).

| User phrase | Pass |
|---|---|
| "robust" / "heteroskedasticity-robust" / "White" | `"HC1"` |
| "classical" / "default" / none specified | `"nonrobust"` |
| "clustered by firm" | `{"cluster": df["firm_id"]}` |
| "Newey-West, 4 lags" / HAC | `{"HAC": 4}` |

### For DID functions (linearmodels PanelOLS-based)

Parameter name: `cov_type`. **Completely different string values** — do NOT pass `"HC1"` or dict here.

| User phrase | Pass |
|---|---|
| "classical" / "default" / none specified | `"unadjusted"` |
| "robust" / "heteroskedasticity-robust" | `"robust"` |
| "clustered by entity / firm / individual" | `"cluster_entity"` |
| "clustered by time / year / period" | `"cluster_time"` |
| "two-way clustered" | `"cluster_both"` |

When the user is vague ("use robust errors on a panel"), pick `"cluster_entity"` for DID — that's the usual applied-econ default (Bertrand, Duflo & Mullainathan 2004).

## Common pitfalls (address proactively)

1. **PSM without checking common support** — Always run `propensity_score_visualize_propensity_score_distribution` before matching; if the overlap is poor, warn the user and suggest trimming.
2. **DID without checking parallel trends** — For staggered DID, run the event study (`Staggered_Diff_in_Diff_Event_Study_regression`) and look at the pre-period coefficients; they should be flat near zero.
3. **IV with a weak instrument** — Always call `IV_2SLS_IV_setting_test` and report the covariate-adjusted partial first-stage F-stat; the rule of thumb is F > 10.
4. **RDD with the wrong bandwidth** — The default bandwidth choice dominates results. Offer at least two bandwidths and check sensitivity.
5. **Fuzzy RDD as a bare ratio** — Use the default `target_type="summary"` so the answer includes the Wald LATE, first-stage jump, approximate SE/CI, within-bandwidth N, and weak-first-stage flag. Use `target_type="estimator"` only for legacy scalar code.
6. **Binary-outcome OLS** — Linear Probability Models are OK for ATE but warn about predicted probabilities outside [0, 1] and offer Logit/Probit as a sensitivity check.

## Known library limitations (tell the user upfront)

These are issues in `lib/econometric_algorithm.py` that you cannot fix by calling the functions differently — they're in the implementation. Disclose them when relevant so the user can judge how much to trust the output.

### IV 2SLS standard errors are biased downward

The library implements 2SLS by hand with two OLS calls. The second-stage OLS computes SEs using the predicted T-hat residuals rather than the actual T residuals. Proper 2SLS SE adjustment (Wooldridge Ch. 15) is not applied. **Reported p-values and CIs will be too optimistic.** For a reliable IV analysis, recommend using `linearmodels.iv.IV2SLS` directly or the R `ivreg` package as a cross-check.

### IV diagnostic checks do not prove exclusion

`IV_2SLS_IV_setting_test` now reports a first stage conditional on covariates, a reduced form, and a residual falsification check. Do **not** call the residual check an exclusion-restriction test. Exclusion is not directly testable in a just-identified IV design; it must be defended with institutional detail, predetermined-covariate balance, placebo outcomes, and sensitivity discussion.

### AIPW is not doubly robust

The implementation only constructs counterfactuals for the control group (predicting Y(1) for controls) but not for the treated group (predicting Y(0) for treated). A proper AIPW (Robins-Rotnitzky-Zhao 1994) requires both directions. The result is asymmetric and **not actually doubly robust**. Prefer PSM + IPW as primary estimates; if true double robustness is needed, use `econml.dr.DRLearner` or `dowhy`.

### IPW-RA takes sqrt of IPW weights

The IPW-RA implementation applies `IPW = IPW ** 0.5` before running the weighted regression. This has no standard theoretical justification in the DR literature (Bang & Robins 2005). Document as non-standard when reporting results.

### Event study lead/lag assignment relies on DataFrame integer index

The code uses `each_index - policy_time_index` where these are pandas integer indices, not actual time-period distances. This works only if the panel is balanced, sorted by time, and has no gaps in the integer index. Filtered or unbalanced panels may get incorrect lead/lag assignments. Always ensure the panel is balanced and sorted before calling the event study function.

## When to break out of the canned methods

The 17 functions cover the common cases, but some requests need custom code — e.g., synthetic control, triple-diff, quantile treatment effects, machine-learning-based heterogeneous effects (Causal Forest, DML). When the user asks for those, tell them honestly that this skill doesn't cover it, and offer to write it from scratch using statsmodels / linearmodels / scikit-learn directly.

## Reference files

- `references/method_selection.md` — decision guide for picking the right estimator
- `references/applied_project_workflow.md` — end-to-end empirical project workflow from question to final report
- `references/advanced_applied_workflow.md` — advanced applied workflow: estimands, identification memos, diagnostics, robustness, heterogeneity
- `references/data_preprocessing_advice.md` — automatic dataset diagnostics and cleaning advice workflow
- `references/diagnostic_checklist.md` — method-specific checks before presenting estimates as causal
- `references/result_tables.md` — compact model comparison table workflow using `lib/result_tables.py`
- `references/method_details.md` — exact signatures and minimal code per function
- `references/interpretation.md` — how to report results for each method family
- `examples/` — runnable end-to-end examples per method family
