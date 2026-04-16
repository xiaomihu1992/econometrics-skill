"""
OLS with covariates — minimal runnable example.

Scenario: Estimate effect of years of schooling (T) on log wages (Y),
controlling for experience and experience^2. Simulated data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import ordinary_least_square_regression

rng = np.random.default_rng(42)
n = 1000
experience = rng.uniform(0, 40, n)
schooling = rng.integers(8, 20, n).astype(float)
# True model: log wage = 1 + 0.08 * schooling + 0.04 * exp - 0.0005 * exp^2 + noise
log_wage = 1 + 0.08 * schooling + 0.04 * experience - 0.0005 * experience ** 2 + rng.normal(0, 0.3, n)

df = pd.DataFrame({
    "log_wage": log_wage,
    "schooling": schooling,
    "experience": experience,
    "experience_sq": experience ** 2,
})

model = ordinary_least_square_regression(
    dependent_variable=df["log_wage"],
    treatment_variable=df["schooling"],
    covariate_variables=df[["experience", "experience_sq"]],
    cov_info="HC1",  # robust SE
)

print(model.summary())
print(f"\nEstimated return to schooling: {model.params['schooling']:.4f} log points")
print(f"  ≈ {np.exp(model.params['schooling']) - 1:.2%} per additional year of schooling")
