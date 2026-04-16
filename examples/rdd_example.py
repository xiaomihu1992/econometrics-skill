"""
Sharp RDD with bandwidth sensitivity.

Scenario: Merit scholarship awarded if entry_exam >= 70.
Effect on college GPA four years later.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import Sharp_Regression_Discontinuity_Design_regression

rng = np.random.default_rng(17)
n = 2500
entry_exam = rng.uniform(40, 100, n)
scholarship = (entry_exam >= 70).astype(float)

# True RDD effect: scholarship lifts GPA by 0.3 points at the cutoff
# Smooth trend in running variable on both sides
gpa = 2.0 + 0.02 * (entry_exam - 70) + 0.3 * scholarship + rng.normal(0, 0.25, n)

df = pd.DataFrame({"gpa": gpa, "scholarship": scholarship, "entry_exam": entry_exam})

cutoff = 70.0
print(f"True effect at cutoff: 0.300\n")

for bw in [5, 10, 20]:
    m = Sharp_Regression_Discontinuity_Design_regression(
        dependent_variable=df["gpa"],
        entity_treatment_dummy=df["scholarship"],
        running_variable=df["entry_exam"],
        covariate_variables=None,
        running_variable_cutoff=cutoff,
        running_variable_bandwidth=float(bw),
        kernel_choice="triangle",
        cov_info="HC1",
    )
    ate = m.params["scholarship"]
    se = m.bse["scholarship"]
    n_used = int(m.nobs)
    print(f"Bandwidth = ±{bw:>3d}  |  ATE = {ate:+.3f}  (SE {se:.3f})  |  n = {n_used}")
