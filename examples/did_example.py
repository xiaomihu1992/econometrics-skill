"""
Static DID with two-way fixed effects and clustered standard errors.

Scenario: Minimum wage increase in some states (treated) in 2015.
Panel of state-year employment rates. Simulated data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import Static_Diff_in_Diff_regression

rng = np.random.default_rng(13)
n_states = 50
years = list(range(2010, 2020))
treated_states = rng.choice(range(n_states), size=20, replace=False)

rows = []
for s in range(n_states):
    state_fe = rng.normal(0, 1)
    for y in years:
        treated = int(s in treated_states)
        post = int(y >= 2015)
        # True DID effect: -0.5 percentage points on employment
        employment = 60 + state_fe + 0.1 * (y - 2010) + (-0.5) * treated * post + rng.normal(0, 0.5)
        rows.append((s, y, employment, treated, post, rng.normal(0, 1)))

df = pd.DataFrame(rows, columns=["state", "year", "employment", "treated_state", "post_2015", "X1"])
df = df.set_index(["state", "year"])

m = Static_Diff_in_Diff_regression(
    dependent_variable=df["employment"],
    treatment_entity_dummy=df["treated_state"],
    treatment_finished_dummy=df["post_2015"],
    covariate_variables=df[["X1"]],
    entity_effect=True,
    time_effect=True,
    cov_type="cluster_entity",
)
print(m)
ate = m.params["treatment_group_treated"]
print(f"\nTrue effect: -0.500 pp")
print(f"DID estimate: {ate:.3f} pp")
