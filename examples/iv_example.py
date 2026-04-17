"""
IV / 2SLS with first-stage diagnostics.

Scenario: Effect of college attendance (T) on earnings (Y), instrumenting
with distance to nearest college (Z). Classic Card (1995) setup, simulated.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import IV_2SLS_regression, IV_2SLS_IV_setting_test

rng = np.random.default_rng(11)
n = 3000
# Unobserved ability affects both college attendance and earnings
ability = rng.normal(0, 1, n)
# Distance to college — the instrument (negatively affects attendance, no direct effect on Y)
distance = rng.exponential(15, n)
# Covariate: parental education
parent_educ = rng.normal(12, 3, n)

# First stage: P(T=1) increases in ability and parent_educ, decreases in distance
logit_t = -1 + 0.8 * ability + 0.1 * parent_educ - 0.05 * distance
p_t = 1 / (1 + np.exp(-logit_t))
T = (rng.uniform(size=n) < p_t).astype(float)

# True causal effect of college on earnings: 0.3 log points
Y = 10 + 0.3 * T + 0.5 * ability + 0.05 * parent_educ + rng.normal(0, 0.5, n)

df = pd.DataFrame({"Y": Y, "T": T, "Z": distance, "parent_educ": parent_educ})

# First-stage / reduced-form / falsification diagnostics
diag = IV_2SLS_IV_setting_test(
    dependent_variable=df["Y"],
    treatment_variable=df["T"],
    IV_variable=df["Z"],
    covariate_variables=df[["parent_educ"]],
)
first_stage = diag["first_stage"]
reduced_form = diag["reduced_form"]
resid_check = diag["residual_falsification_check"]
print("First-stage diagnostics (relevance, conditional on covariates):")
print(f"  Partial F-statistic = {diag['first_stage_partial_f']:.2f} (p = {diag['first_stage_partial_f_pvalue']:.4f})")
print(f"  R² = {first_stage.rsquared:.4f}")
print(f"  --> {'Strong' if diag['first_stage_partial_f'] > 10 else 'WEAK'} instrument (threshold: F > 10)")
print()
print("Reduced form (Y on Z + covariates):")
print(f"  Coeff on Z = {reduced_form.params['Z']:.4f} (p = {reduced_form.pvalues['Z']:.4f})")
print()
print("Residual falsification check (not an exclusion-restriction test):")
print(f"  Coeff on Z = {resid_check.params['Z']:.4f} (p = {resid_check.pvalues['Z']:.4f})")
print("  Exclusion itself must be defended from institutional details and placebo/balance checks.")
print()

# 2SLS estimate (LATE)
m = IV_2SLS_regression(
    dependent_variable=df["Y"],
    treatment_variable=df["T"],
    IV_variable=df["Z"],
    covariate_variables=df[["parent_educ"]],
    cov_info="HC1",
)
ate = m.params["T"]
se = m.bse["T"]
print(f"2SLS LATE estimate: {ate:.3f} (SE = {se:.3f})")
print(f"True effect: 0.300")
print()
print("CAVEAT: This library implements manual 2SLS; second-stage SEs are not")
print("adjusted for the first-stage estimation and may be too small.")
print("For publication, cross-check with linearmodels.iv.IV2SLS or R's ivreg.")
