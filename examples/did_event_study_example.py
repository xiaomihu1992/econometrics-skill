"""
Staggered DID with Event Study and visualization.

Scenario: States adopt a clean-air regulation at different times (staggered rollout).
Panel of state-year pollution levels (PM2.5). Some states never adopt.
We estimate the dynamic treatment effect and check parallel trends.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pandas as pd
from econometric_algorithm import (
    Staggered_Diff_in_Diff_regression,
    Staggered_Diff_in_Diff_Event_Study_regression,
    Staggered_Diff_in_Diff_Event_Study_visualization,
)

rng = np.random.default_rng(42)
n_states = 40
years = list(range(2005, 2021))  # 16 years

# 15 states adopt the regulation at different times; 25 never adopt
adopt_times = {s: rng.choice(range(2010, 2016)) for s in range(15)}

rows = []
for s in range(n_states):
    state_fe = rng.normal(50, 5)  # baseline pollution varies by state
    for y in years:
        time_trend = -0.3 * (y - 2005)  # general downward trend in pollution
        treated = 0
        if s in adopt_times and y >= adopt_times[s]:
            treated = 1
        # True effect: -2.0 units of PM2.5 upon adoption (instant, persistent)
        pm25 = state_fe + time_trend + (-2.0) * treated + rng.normal(0, 1.5)
        rows.append((s, y, pm25, treated))

df = pd.DataFrame(rows, columns=["state_id", "year", "pm25", "treated"])
df = df.set_index(["state_id", "year"])

# ============================================================
# 1) Staggered DID — overall treatment effect
# ============================================================
m_did = Staggered_Diff_in_Diff_regression(
    dependent_variable=df["pm25"],
    entity_treatment_dummy=df["treated"],
    covariate_variables=None,
    entity_effect=True,
    time_effect=True,
    cov_type="cluster_entity",
)
print("=" * 60)
print("Staggered DID — TWFE estimate")
print("=" * 60)
print(f"Coefficient (treatment_entity_treated): {m_did.params['treatment_entity_treated']:.3f}")
print(f"P-value: {m_did.pvalues['treatment_entity_treated']:.4f}")
print(f"True effect: -2.000")
print()

# ============================================================
# 2) Event Study — check parallel trends + dynamic effects
# ============================================================
# Need see_back_length >= 4, see_forward_length >= 3
m_es = Staggered_Diff_in_Diff_Event_Study_regression(
    dependent_variable=df["pm25"],
    entity_treatment_dummy=df["treated"],
    covariate_variables=None,
    see_back_length=4,
    see_forward_length=4,
    entity_effect=True,
    time_effect=True,
    cov_type="cluster_entity",
)

print("=" * 60)
print("Event Study — lead/lag coefficients")
print("=" * 60)
# Coefficient names: Lead_D4+, Lead_D3, Lead_D2, [Lead_D1 omitted], D0, Lag_D1, ..., Lag_D4+
coef_names = ["Lead_D4+", "Lead_D3", "Lead_D2", "D0", "Lag_D1", "Lag_D2", "Lag_D3", "Lag_D4+"]
for name in coef_names:
    if name in m_es.params.index:
        coef = m_es.params[name]
        pval = m_es.pvalues[name]
        print(f"  {name:>10s}: coeff = {coef:+.3f}  (p = {pval:.3f})")

print()
print("Interpretation:")
print("  - Lead coefficients (Lead_D4+ to Lead_D2) should be near zero = parallel trends hold")
print("  - D0 and Lag coefficients show the treatment effect dynamics")

# ============================================================
# 3) Visualization
# ============================================================
fig = Staggered_Diff_in_Diff_Event_Study_visualization(
    regression_model=m_es,
    see_back_length=4,
    see_forward_length=4,
)
fig.savefig("event_study_plot.png", dpi=150, bbox_inches="tight")
print("\nEvent study plot saved to event_study_plot.png")
