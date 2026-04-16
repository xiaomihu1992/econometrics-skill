# Diagnostic Checklist

Use this checklist before presenting an estimate as causal. If an item fails, report it plainly and downgrade the strength of the conclusion.

## Universal checks

- Treatment, outcome, controls, unit, and time variables are named and typed correctly.
- Treatment is measured before the outcome.
- Controls are pre-treatment and not mediators, colliders, or mechanically affected by treatment.
- Missing values are handled explicitly.
- Categorical variables are encoded intentionally.
- Outliers and influential observations are checked.
- Standard errors are clustered at the assignment or shock-correlation level when applicable.
- The target population after trimming, matching, or sample restrictions is stated.

## OLS with controls

- Explain why conditional independence might be plausible.
- Show a specification ladder: no controls, baseline controls, richer controls, fixed effects if relevant.
- Check functional form for key continuous controls.
- Check leverage and influential observations.
- Avoid interpreting R-squared as evidence of causal credibility.

## Propensity score methods

- Plot propensity score overlap before matching or weighting.
- Report standardized mean differences before and after adjustment.
- Inspect extreme IPW weights.
- Try trimming or calipers when overlap is weak.
- State whether the estimand after trimming is still the original target.
- Treat non-standard AIPW/IPW-RA functions in this library as sensitivity checks, not primary evidence.

## IV / 2SLS

- Report first-stage coefficient on the instrument conditional on covariates.
- Report partial F-statistic or an equivalent relevance diagnostic.
- Report reduced-form effect of the instrument on the outcome.
- Check whether the instrument predicts predetermined covariates.
- Explain exclusion restriction and monotonicity in institutional terms.
- If the instrument is weak, avoid strong causal language.
- For serious work, prefer `linearmodels.iv.IV2SLS` over the bundled hand-rolled 2SLS.

## DID and event studies

- Define treated cohorts, never-treated units, and treatment timing.
- Plot raw group or cohort trends before regression.
- Estimate event-study leads and inspect them before interpreting lags.
- State the omitted event-time period.
- Check anticipation by excluding periods immediately before treatment.
- Check sensitivity to balanced panels, sample windows, and excluding early/late cohorts.
- Warn about TWFE bias under staggered adoption with heterogeneous effects.

## RDD

- Plot outcome against the running variable near the cutoff.
- Plot or inspect the running-variable density near the cutoff.
- Test predetermined covariate continuity at the cutoff.
- Report bandwidth sensitivity: smaller, preferred, larger.
- Prefer local linear specifications as primary evidence.
- For fuzzy RDD, report the first-stage treatment jump at the cutoff.
- Consider donut RDD when manipulation right at the cutoff is plausible.

## Heterogeneity

- Subgroups are defined before observing results when possible.
- Subgroup variables are pre-treatment.
- Sample size and overlap are adequate within each subgroup.
- Use interactions when comparing subgroup effects statistically.
- Warn about multiple testing when many groups are explored.

## Red flags

- Treatment effect appears before treatment.
- Results depend on one arbitrary bandwidth, window, or control set.
- Significance disappears under appropriate clustering.
- Placebo outcomes show effects similar to the main outcome.
- The treatment changes who appears in the sample.
- The proposed control variable is downstream of treatment.
