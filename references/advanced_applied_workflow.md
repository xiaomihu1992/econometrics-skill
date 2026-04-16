# Advanced Applied Econometrics Workflow

Load this reference when the user wants referee-response, advanced research, or publication-grade causal analysis. The goal is to move from "run an estimator" to "defend a research design."

## Output contract

For advanced work, structure the answer as a research memo:

1. **Causal question**: one precise sentence.
2. **Estimand**: ATE, ATT, LATE, ITT, TOT, dynamic effect, cohort-specific effect, or dose-response; include target population and time horizon.
3. **Identification strategy**: what variation identifies the effect, why naive comparisons are biased, and which assumption is doing the work.
4. **Preferred specification**: equation, sample, fixed effects, controls, clustering, and treatment timing/cutoff/instrument definitions.
5. **Diagnostics**: method-specific validity checks.
6. **Robustness grid**: what changes, what should remain stable, and what would invalidate the result.
7. **Interpretation**: magnitude, uncertainty, economic significance, and external validity.
8. **Threat model**: remaining untestable assumptions and stronger designs or data that would reduce them.

## Estimand discipline

Do not let the estimator define the estimand accidentally. Force clarity first:

| User goal | Estimand | Notes |
|---|---|---|
| Average policy effect on all eligible units | ATE | Requires credible counterfactual for treated and untreated units |
| Effect on participants or treated units | ATT | Natural for matching and many program evaluations |
| Effect induced by an instrument | LATE | IV identifies compliers, not necessarily the whole population |
| Effect of assignment/eligibility | ITT | Important for lotteries, RDD assignment, or imperfect compliance |
| Effect of treatment among those induced by assignment | TOT / treatment-on-treated | Often ITT divided by first-stage take-up under strong assumptions |
| Event-time effect after rollout | Dynamic treatment effects | Needs time horizon, omitted event period, and anticipation assumptions |
| Effect by subgroup | Heterogeneous treatment effect | Pre-specify groups; treat data-mined groups as exploratory |

Always state the outcome scale: levels, logs, percentage points, standardized units, or binary linear probability. For log outcomes, translate approximate percent effects carefully.

## Design before regression

Before fitting models, write a short identification memo:

- **Treatment assignment process**: who gets treated, when, and why.
- **Counterfactual comparison**: which units approximate untreated potential outcomes.
- **Bad controls**: exclude post-treatment variables, mediators, colliders, and variables mechanically affected by treatment.
- **Sample construction**: inclusion/exclusion rules, attrition, missingness, panel balance, and whether treatment affects sample selection.
- **Timing**: define pre-period, treatment onset, anticipation window, and outcome measurement window.
- **Interference/SUTVA**: consider spillovers, displacement, equilibrium effects, and geographic contamination.
- **Measurement**: treatment misclassification, outcome measurement error, top-coding, winsorization, and survey weights.

If the identifying variation cannot be stated in one paragraph, pause and refine the design before estimating.

## OLS with controls

Treat OLS as a design-based estimator only under a credible conditional independence story. For advanced work:

- Report a specification ladder: no controls, baseline controls, rich pre-treatment controls, fixed effects, and the preferred model.
- Justify each control as pre-treatment. Do not include mediators just because they improve R-squared.
- Consider nonlinearities and interactions when theory suggests them; do not rely on a single linear control for strong confounders.
- Choose clustering at the level of treatment assignment or shock correlation, not the level that gives the smallest SE.
- Check influential observations, leverage, functional form, and whether a small number of treated units drives results.
- If omitted-variable bias is the main threat, discuss sensitivity rather than pretending controls solve it.

## Propensity score designs

Use propensity scores to make treatment and control groups comparable on observed pre-treatment covariates, not as a mechanical black box.

Advanced diagnostics:

- Plot treated/control propensity score distributions and report common support.
- Report standardized mean differences before and after matching/weighting.
- Use trimming or calipers when overlap is poor; state the resulting target population.
- Inspect extreme IPW weights; consider stabilized weights and weight truncation as sensitivity checks.
- Re-estimate after dropping observations with propensity scores near 0 or 1.
- Prefer PSM and IPW as primary estimates in this library. Treat the AIPW/IPW-RA functions here as non-standard sensitivity checks.

If the user needs publication-grade double robustness, write custom code or use a dedicated package such as `econml` or `dowhy` when available.

## IV / 2SLS

For IV, the real work is defending the instrument, not running two stages.

Required reporting:

- First stage conditional on covariates, with coefficient on Z, partial F-statistic, and interpretation.
- Reduced form: effect of Z on Y.
- 2SLS/LATE estimate: effect for compliers, not the population ATE.
- Exclusion restriction argument: why Z affects Y only through T.
- Monotonicity argument: why Z does not push some units into treatment and others out in opposite directions.
- Balance/falsification checks: Z should not predict predetermined covariates or placebo outcomes.
- Weak-instrument concern: if first-stage F is weak, do not trust conventional SEs.

The bundled `IV_2SLS_regression` is useful for quick checks, but its second-stage SEs are biased downward. For serious work, use `linearmodels.iv.IV2SLS` directly and report robust or clustered SEs.

## Difference-in-differences and event studies

DID is a research design around counterfactual trends, not just an interaction term.

Advanced workflow:

- Define treatment cohorts, never-treated units, already-treated units, and anticipation periods.
- Plot raw means by group/cohort before regression.
- Estimate an event study and interpret leads before lags.
- State the omitted event period and whether effects are relative to `t=-1`.
- Cluster at the treatment assignment level; with few clusters, warn about inference fragility.
- Check composition changes, attrition, differential missingness, and treatment-induced sample selection.
- Run placebo treatment dates in pre-periods when enough pre-data exist.
- Run sensitivity to excluding early/late cohorts and to balanced vs. unbalanced panels.

For staggered adoption with heterogeneous effects, two-way fixed effects can be misleading because already-treated units may act as controls for newly treated units. For publication-grade work, mention Sun-Abraham or Callaway-Sant'Anna style estimators and implement them separately when required.

## Regression discontinuity

RDD credibility comes from local quasi-random assignment near the cutoff.

Advanced workflow:

- Show the running-variable distribution and outcome bins around the cutoff.
- Use local linear specifications; avoid high-order global polynomials as primary evidence.
- Prefer triangular kernel for the main local specification unless there is a reason not to.
- Report bandwidth sensitivity: half, preferred, double; include sample size within bandwidth.
- Test covariate continuity at the cutoff.
- Check density/manipulation near the cutoff; discuss McCrary-style evidence when possible.
- Consider donut RDD if precise manipulation or sorting is plausible right at the cutoff.
- For fuzzy RDD, report the first-stage jump in treatment probability and interpret the result as LATE for compliers near the cutoff.

## Robustness grid

A strong applied answer includes a table like this:

| Dimension | Baseline | Robustness variants | Failure signal |
|---|---|---|---|
| Controls | Preferred pre-treatment controls | sparse/rich controls, nonlinear terms | sign/magnitude flips without explanation |
| Sample | Main analytic sample | balanced panel, trimmed sample, exclude outliers | effect driven by one restriction |
| Inference | preferred clustering | alternative cluster level, robust SE | significance only under convenient SE |
| Timing | preferred window | alternative pre/post windows, anticipation exclusion | effect appears before treatment |
| Functional form | levels/logs as justified | alternative transformations | result exists only on one arbitrary scale |
| Placebo | none or stated | placebo outcome, placebo cutoff/date/group | placebo effects similar to main effect |

Do not call a result robust merely because p-values stay below 0.05. Robustness is about estimand stability and design credibility.

## Heterogeneity

Heterogeneity can be substantive or noise. For advanced work:

- Pre-specify subgroups based on theory or institutions.
- Use interactions rather than separate regressions when comparing effects statistically.
- Report subgroup sample sizes and overlap/support.
- Treat post-treatment subgroup variables as bad controls unless the estimand is explicitly conditional on post-treatment status.
- Warn about multiple testing when many subgroups are explored.
- Distinguish effect heterogeneity from selection into treatment intensity or timing.

## Final writing standard

A publication-grade applied conclusion should not sound like "the coefficient is significant." It should say:

- what the estimand is;
- why the design identifies it;
- how large the effect is in meaningful units;
- what diagnostics support or weaken the design;
- which robustness checks matter most;
- what remains untestable;
- whether the evidence is causal, suggestive, or only descriptive.
