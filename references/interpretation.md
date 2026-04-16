# Results Interpretation Guide

A regression table is not an answer. Your job is to turn coefficients into a paragraph the user can put in a slide, memo, or final report. Follow the templates below per method family.

## General reporting structure

Regardless of method, hit these five points in this order:

1. **The causal question** (one sentence) — "Does X cause Y?"
2. **Identification strategy** (one sentence) — "We identify the effect using [method] under the assumption that [key assumption]."
3. **Point estimate & units** — "A one-unit increase in X is associated with a [sign][magnitude] [unit] change in Y."
4. **Uncertainty** — standard error, 95% CI, p-value. Don't just say "significant" — state the CI.
5. **Caveats / diagnostics** — which assumption is most fragile, what robustness checks pass, what doesn't.

---

## OLS

**Template**:
> Controlling for [covariates], a one-unit increase in T is associated with a [β] [unit of Y] change in Y (SE = [se], 95% CI [low, high], p = [p]). This is a descriptive association; for a causal interpretation we would need [randomization / a credible natural experiment / selection on observables].

**If the user is using OLS as a causal estimator (claiming CIA holds on observed covariates)**, flag that this is a strong assumption and suggest PS methods (PSM + IPW as a consistency check) as a more flexible alternative.

---

## Propensity Score methods (PSM / IPW / AIPW / IPW-RA)

Always report **three things together**:

1. The ATE (or ATT, whichever was requested — note: this library supports ATE and ATT only, not ATC)
2. **Overlap diagnostic** — reference the distribution plot; state whether common support holds
3. **Balance** (optional but best-in-class) — standardized differences in covariates between treated/control, before vs. after weighting/matching; good work shows the reduction

**Template** (for PSM / IPW, which return point estimates only):
> Using [PSM with 1-NN matching / IPW], we estimate an ATE of [β] [unit of Y]. Bootstrap standard error: SE = [se], 95% CI = [[low], [high]] (based on [B] bootstrap replications, resampling the full PS estimation + matching pipeline). The propensity score overlap plot shows [good / mediocre / poor] common support over ê ∈ [0.1, 0.9]. The key identifying assumption is unconfoundedness conditional on [X1, X2, ...]; violations (unobserved confounders correlated with both T and Y) would bias this estimate in an unknown direction.

**For PS-regression / IPW-RA** (which return statsmodels model objects with built-in SEs):
> Using [PS regression / IPW-RA], we estimate an ATE of [β] [unit of Y] (SE = [se], 95% CI [low, high], p = [p]).

**When the user gave ambiguous instructions**, default to running **both PSM and IPW** and compare — if they agree, the estimate is robust. If they diverge substantially, investigate overlap and weight diagnostics before choosing. AIPW/IPW-RA serve as additional sensitivity checks, but note the AIPW implementation caveat (see method_details.md §6).

---

## IV / 2SLS

**Never** report a 2SLS estimate without the first-stage F-statistic.

**Template**:
> Instrumenting T with Z, the 2SLS estimate of the local average treatment effect (LATE) is [β] [unit of Y] (SE = [se], 95% CI [low, high], p = [p]). The first-stage F-statistic is [F]; [values above 10 suggest the instrument is not weak / values below 10 suggest weak-instrument bias is a concern]. Identification requires that Z affects Y only through T (exclusion restriction) and is as-good-as-random conditional on covariates — assumptions that cannot be fully tested and should be defended on institutional grounds.

**LATE vs. ATE**: 2SLS identifies the LATE — the effect on "compliers", i.e. units whose treatment status is changed by the instrument. This may differ from the population ATE. Mention this if the user is extrapolating.

---

## DID (Static)

**Template**:
> Using difference-in-differences, comparing [treated group] to [control group] before and after [event / policy], we estimate a treatment effect of [β] [unit of Y] (SE = [se], 95% CI [low, high], p = [p]). Standard errors are clustered by [entity / unit] to account for serial correlation. The key identifying assumption is parallel trends: absent the policy, treated and control units would have evolved similarly in Y. [The event-study plot / pre-trend test] shows [flat / diverging] pre-trends, supporting / challenging this assumption.

---

## DID (Staggered)

Plus the above, add:

> Because treatment timing varies across units, we estimate a two-way fixed effects model. Note: TWFE can be biased when treatment effects are heterogeneous across cohorts (Goodman-Bacon 2021); a robustness check using a heterogeneity-robust estimator (Callaway-Sant'Anna or Sun-Abraham) would strengthen the conclusion.

---

## DID Event Study

**Template**:
> The event-study plot traces out the dynamic effect of treatment. Lead coefficients (pre-periods, k < 0) are [flat near zero / trending], [supporting / undermining] the parallel-trends assumption. The effect [appears on impact at k=0 / builds up over k = 1, 2, 3 / attenuates after k = 2]. At the peak (k = [k*]), the effect is [β] [unit of Y] (95% CI [low, high]).

**Always interpret the leads before the lags.** If leads are trending, the lags are not interpretable as causal — say so.

---

## RDD (Sharp)

**Template**:
> At the cutoff [c = ...], crossing from just below to just above is associated with a [β] [unit of Y] jump in Y (SE = [se], 95% CI [low, high], p = [p]). We use a local linear regression with a [triangle / uniform] kernel and bandwidth of [h]. The estimate is robust to halving and doubling the bandwidth: [β/2h = ..., β/2h = ...]. The key identifying assumption is that units could not precisely manipulate their running variable around the cutoff — [a McCrary density test / a visual check of the density at c] [supports / does not support] this.

Always report **at least two bandwidths** — the chosen one and at least one alternative.

---

## RDD (Fuzzy)

Plus the above, add:

> Because crossing the cutoff does not deterministically assign treatment, we use a fuzzy RDD, instrumenting actual treatment status with the threshold-crossing indicator. The first-stage jump in treatment probability at the cutoff is [π] ([F-stat]). The second-stage LATE — the effect on units induced into treatment by the threshold — is [β].

---

## When results are "insignificant"

Don't say "no effect". Say:

> The point estimate is [β] [unit], with a 95% CI of [low, high]. We cannot distinguish this from zero at conventional levels (p = [p]), but the CI does not rule out economically meaningful effects up to [max absolute CI]. With the current sample size, the minimum detectable effect at 80% power is approximately [...].

An imprecise null is different from a precise null.

---

## When diagnostics fail

Be direct. Example phrasings:

- **Poor PS overlap**: "The propensity score distributions for treated and control units overlap only in a narrow region. Outside that region, we cannot construct credible counterfactuals. We recommend restricting the analysis to the region of common support or acknowledging that the estimated ATE applies only to a subset of the population."
- **Weak IV (F < 10)**: "The first-stage F-statistic is [F], below the conventional threshold of 10. The 2SLS estimate is therefore subject to weak-instrument bias in an unknown direction, and confidence intervals based on standard asymptotics are unreliable. We recommend treating the estimate as suggestive rather than definitive, or finding a stronger instrument."
- **Pre-trend fails**: "The pre-period event-study coefficients trend [upward / downward], inconsistent with parallel trends. The DID estimate is therefore not cleanly interpretable as a causal effect. We could attempt to adjust for pre-trends, use a different comparison group, or switch identification strategies."
- **Bandwidth sensitivity**: "The RDD estimate changes sign when the bandwidth is halved, from [β1] to [β2]. This suggests the local linear fit is sensitive to the functional form near the cutoff and the estimate should not be reported without further investigation."

Honest reporting of what doesn't work is more valuable than polished reporting of what might not hold.
