# Applied Project Workflow

Use this reference when the user needs end-to-end guidance for an empirical project, not just one estimator. Keep the language practical: move from question to data, design, estimation, diagnostics, robustness, and final reporting.

## Workflow overview

1. **Research question**: State one causal question in terms of treatment, outcome, population, and period.
2. **Variable map**: Define outcome, treatment, unit, time, controls, subgroup variables, and sample filters.
3. **Estimand**: Choose ATE, ATT, LATE, ITT, dynamic effect, or subgroup effect.
4. **Identification strategy**: Explain the source of identifying variation and why naive comparisons are biased.
5. **Data audit**: Check missingness, duplicates, outliers, panel balance, treatment timing, and post-treatment variables.
6. **Baseline specification**: Write the preferred equation and justify fixed effects, controls, and standard errors.
7. **Diagnostics**: Run method-specific checks before interpreting results.
8. **Robustness plan**: Define planned alternative specifications before looking for convenient results.
9. **Heterogeneity and mechanisms**: Separate pre-specified subgroup analysis from exploratory patterns.
10. **Final report**: Present estimates, uncertainty, economic magnitude, diagnostics, robustness, and remaining threats.

## Variable map template

| Role | Variable | Required checks |
|---|---|---|
| Outcome | `Y` | scale, units, missingness, outliers, timing relative to treatment |
| Treatment | `T` | binary/continuous, timing, intensity, compliance, anticipation |
| Unit id | `id` | duplicates, stable identifiers, treatment assignment level |
| Time | `time` | sorting, gaps, frequency, pre/post coverage |
| Controls | `X` | pre-treatment only, missingness, transformations |
| Cluster | `cluster_id` | should match assignment or shock correlation level |
| Subgroups | `G` | pre-treatment status, adequate sample size, overlap |

## Analysis sequence

Use this order unless the design demands otherwise:

1. Load data with `load_table()` and inspect columns with `get_column_info()`.
2. Clean identifiers, dates, treatment indicators, and categorical controls.
3. Produce descriptive counts: units, periods, treated units, control units, missingness.
4. State the estimand and preferred design before running estimates.
5. Run the baseline model.
6. Run diagnostics that test observable implications of the design.
7. Run robustness variants from a pre-specified grid.
8. Interpret magnitude in units the user can understand.
9. State what remains untestable and what better data/design would improve credibility.

## Final report skeleton

```text
Question:
Estimand:
Data and sample:
Identification strategy:
Preferred specification:
Main estimate:
Diagnostics:
Robustness:
Heterogeneity or mechanisms:
Limitations:
Next steps:
```

Do not let output become a model dump. The central object is the credibility of the research design.
