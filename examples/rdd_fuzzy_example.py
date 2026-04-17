"""
Fuzzy RDD — two-step local linear and global polynomial.

Scenario: A municipality offers a housing subsidy to families scoring above
a threshold on a poverty index. But take-up is imperfect (fuzzy): crossing
the cutoff raises the probability of receiving the subsidy from ~20% to ~70%.
We estimate the effect of the subsidy on housing quality (continuous outcome).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import (
    Fuzzy_Regression_Discontinuity_Design_regression,
    Fuzzy_RDD_Global_Polynomial_Estimator_regression,
)

rng = np.random.default_rng(23)
n = 3000
cutoff = 50.0

# Running variable: poverty index score (0-100)
poverty_score = rng.uniform(10, 90, n)

# Fuzzy assignment: crossing cutoff raises P(subsidy) but doesn't guarantee it
above = (poverty_score >= cutoff).astype(float)
noise = rng.normal(0, 1, n)
p_subsidy = 0.2 + 0.5 * above + 0.05 * noise  # ~20% below, ~70% above
p_subsidy = np.clip(p_subsidy, 0.01, 0.99)
subsidy = (rng.uniform(size=n) < p_subsidy).astype(float)

# True LATE (effect on compliers): +5 units of housing quality
housing_quality = 30 + 5.0 * subsidy + 0.1 * (poverty_score - cutoff) + rng.normal(0, 2, n)

df = pd.DataFrame({
    "housing_quality": housing_quality,
    "subsidy": subsidy,
    "poverty_score": poverty_score,
})

# Verify fuzziness: take-up differs across cutoff but isn't deterministic
below_takeup = subsidy[poverty_score < cutoff].mean()
above_takeup = subsidy[poverty_score >= cutoff].mean()
print(f"Take-up below cutoff: {below_takeup:.1%}")
print(f"Take-up above cutoff: {above_takeup:.1%}")
print(f"First-stage jump: {above_takeup - below_takeup:.1%}")
print()

# ============================================================
# 1) Two-step Fuzzy RDD (local linear, bandwidth sensitivity)
# ============================================================
print("=" * 60)
print("Fuzzy RDD — Two-step local linear")
print("=" * 60)
for bw in [8, 15, 25]:
    summary = Fuzzy_Regression_Discontinuity_Design_regression(
        dependent_variable=df["housing_quality"],
        entity_treatment_dummy=df["subsidy"],
        running_variable=df["poverty_score"],
        covariate_variables=None,
        running_variable_cutoff=cutoff,
        running_variable_bandwidth=float(bw),
        kernel_choice="triangle",
        cov_info="HC1",
        target_type="summary",
    )
    weak_note = " [weak first stage]" if summary["weak_first_stage"] else ""
    print(
        f"  Bandwidth ±{bw:>2d}: LATE = {summary['wald_late']:+.3f} "
        f"(approx SE = {summary['approx_delta_se']:.3f}, "
        f"first-stage jump = {summary['first_stage_jump']:.3f}, "
        f"N = {summary['within_bandwidth_n']}){weak_note}"
    )

# Also inspect the preferred-bandwidth summary and fitted models
preferred = Fuzzy_Regression_Discontinuity_Design_regression(
    dependent_variable=df["housing_quality"],
    entity_treatment_dummy=df["subsidy"],
    running_variable=df["poverty_score"],
    covariate_variables=None,
    running_variable_cutoff=cutoff,
    running_variable_bandwidth=15.0,
    kernel_choice="triangle",
    cov_info="HC1",
    target_type="summary",
)
models = preferred["models"]
print(f"\n  Reduced form (Y on above_cutoff):")
print(f"    Coeff on threshold = {preferred['reduced_form_jump']:.3f}")
print(f"  First stage (T on above_cutoff):")
print(f"    Coeff on threshold = {preferred['first_stage_jump']:.3f}")
print(f"  Approx 95% CI for LATE: [{preferred['ci_low']:.3f}, {preferred['ci_high']:.3f}]")
print(f"  SE note: {preferred['se_note']}")

# ============================================================
# 2) Global polynomial Fuzzy RDD (robustness check)
# ============================================================
print()
print("=" * 60)
print("Fuzzy RDD — Global polynomial (robustness)")
print("=" * 60)
for order in [1, 2, 3]:
    m = Fuzzy_RDD_Global_Polynomial_Estimator_regression(
        dependent_variable=df["housing_quality"],
        entity_treatment_dummy=df["subsidy"],
        running_variable=df["poverty_score"],
        covariate_variables=None,
        running_variable_cutoff=cutoff,
        max_order=order,
        kernel_choice="uniform",
        cov_info="HC1",
    )
    ate = m.params["subsidy"]
    se = m.bse["subsidy"]
    print(f"  Order {order}: LATE = {ate:+.3f} (SE = {se:.3f})")

print(f"\nTrue LATE: +5.000")
print("\nNote: Local linear (bandwidth=15) is the preferred specification.")
print("Global polynomial is shown for sensitivity — prefer low orders (1-2).")
print("See Gelman & Imbens (2019) on why high-order global polynomials are unreliable.")
