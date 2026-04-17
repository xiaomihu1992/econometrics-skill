"""
Core econometric algorithm functions for causal inference.

Cleaned from the original MetaGPT tool registration. All functions accept
pandas Series/DataFrame inputs and return results directly.

Dependencies: numpy, pandas, matplotlib, statsmodels, linearmodels, scipy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
import scipy.stats


def _as_dataframe(value, default_name: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        name = value.name or default_name
        return value.rename(name).to_frame()
    return value.copy()


def _f_test_scalar(test_result):
    value = np.asarray(test_result).squeeze()
    return float(value)


def _validate_target_type(target_type: str, allowed: tuple[str, ...], function_name: str) -> None:
    if target_type not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{function_name}: target_type must be one of {allowed_text}; got {target_type!r}.")


def ordinary_least_square_regression(dependent_variable, treatment_variable, covariate_variables, weights=None, cov_info="nonrobust", target_type="final_model", output_tables=False):
    """
    Use Ordinary Least Square Regression method to estimate Average Treatment Effect (ATE) of
    the treatment variable towards the dependent variable.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Target treatment variable.
        covariate_variables (pd.DataFrame or None): Proposed covariate variables.
        weights (pd.Series or None): Weights for WLS. None for standard OLS.
        cov_info (str or dict): Covariance estimator. Supports "nonrobust", "HC0"-"HC3", {"HAC": maxlags}, {"cluster": groups}.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if isinstance(cov_info, str) and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported!")
    elif isinstance(cov_info, dict) and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "ordinary_least_square_regression")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis=1).astype(float)

    if weights is None:
        if isinstance(cov_info, str):
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type=cov_info)
        elif list(cov_info.keys())[0] == "HAC":
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
        elif list(cov_info.keys())[0] == "cluster":
            regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})
    else:
        if isinstance(cov_info, str):
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights=weights).fit(cov_type=cov_info)
        elif list(cov_info.keys())[0] == "HAC":
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights=weights).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
        elif list(cov_info.keys())[0] == "cluster":
            regression = sm.WLS(dependent_variable, sm.add_constant(X), weights=weights).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})

    if output_tables:
        print(regression.summary())

    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared_adj
    elif target_type == "final_model":
        return regression


def propensity_score_construction(treatment_variable, covariate_variables):
    """
    Construct propensity score using binary Logistic regression.

    Args:
        treatment_variable (pd.Series): Binary treatment variable (1=treatment, 0=control).
        covariate_variables (pd.DataFrame): Covariate variables.

    Returns:
        pd.Series: Estimated propensity scores named "propensity_score".
    """
    treatment_variable = treatment_variable.astype(float)
    covariate_variables = covariate_variables.astype(float)

    clf = sm.Logit(treatment_variable, sm.add_constant(covariate_variables).astype(float)).fit(disp=0)
    result_series = pd.Series(clf.predict(sm.add_constant(covariate_variables).astype(float)), index=covariate_variables.index)
    result_series.name = "propensity_score"
    return result_series


def propensity_score_visualize_propensity_score_distribution(treatment_variable, propensity_score):
    """
    Visualize propensity score distribution for treatment and control groups.

    Args:
        treatment_variable (pd.Series): Binary treatment variable.
        propensity_score (pd.Series): Propensity scores.

    Returns:
        matplotlib.figure.Figure: The histogram figure.
    """
    treatment_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
    control_group_propensity = propensity_score.loc[treatment_variable[treatment_variable == 0].index]

    fig, ax = plt.subplots()
    ax.hist(control_group_propensity, bins=40, facecolor="blue", edgecolor="black", alpha=0.7, label="control")
    ax.hist(treatment_group_propensity, bins=40, facecolor="red", edgecolor="black", alpha=0.7, label="treatment")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Propensity Score Distribution")
    ax.legend()
    return fig


def propensity_score_matching(dependent_variable, treatment_variable, propensity_score, matched_num=1, target_type="ATE"):
    """
    Propensity Score Matching (PSM) to estimate ATE or ATT.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Binary treatment variable.
        propensity_score (pd.Series): Propensity scores.
        matched_num (int): Number of nearest neighbors for matching.
        target_type (str): "ATE" or "ATT".
    """
    _validate_target_type(target_type, ("ATE", "ATT"), "propensity_score_matching")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)

    if target_type == "ATE":
        treatment_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
        control_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
        matched_control_dependent_variable_series = pd.Series(index=treatment_group_propensity_score_series.index)
        matched_treatment_dependent_variable_series = pd.Series(index=control_group_propensity_score_series.index)

        for each_index in treatment_group_propensity_score_series.index:
            selected_distance_metric = (control_group_propensity_score_series - treatment_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_control_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()

        for each_index in control_group_propensity_score_series.index:
            selected_distance_metric = (treatment_group_propensity_score_series - control_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_treatment_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()

        ATE = pd.concat([dependent_variable.loc[treatment_group_propensity_score_series.index], matched_treatment_dependent_variable_series]).mean() - pd.concat([dependent_variable.loc[control_group_propensity_score_series.index], matched_control_dependent_variable_series]).mean()
        return ATE

    elif target_type == "ATT":
        treatment_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 1].index]
        control_group_propensity_score_series = propensity_score.loc[treatment_variable[treatment_variable == 0].index]
        matched_dependent_variable_series = pd.Series(index=treatment_group_propensity_score_series.index)
        for each_index in treatment_group_propensity_score_series.index:
            selected_distance_metric = (control_group_propensity_score_series - treatment_group_propensity_score_series.loc[each_index]).map(lambda x: abs(x))
            selected_distance_metric = selected_distance_metric.sort_values()
            selected_index = selected_distance_metric.head(matched_num)
            selected_index = selected_distance_metric[selected_distance_metric.isin(list(selected_index.values))]
            matched_dependent_variable_series.loc[each_index] = dependent_variable.loc[selected_index.index].mean()

        treatment_group_dependent_variable_series = dependent_variable.loc[treatment_group_propensity_score_series.index]
        ATT = treatment_group_dependent_variable_series.mean() - matched_dependent_variable_series.mean()
        return ATT


def propensity_score_inverse_probability_weighting(dependent_variable, treatment_variable, propensity_score, target_type="ATE"):
    """
    Propensity Score Inverse Probability Weighting (IPW) to estimate ATE or ATT.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Binary treatment variable.
        propensity_score (pd.Series): Propensity scores.
        target_type (str): "ATE" or "ATT".
    """
    _validate_target_type(target_type, ("ATE", "ATT"), "propensity_score_inverse_probability_weighting")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)

    if target_type == "ATE":
        ATE_first_part = (dependent_variable * treatment_variable / propensity_score).sum() / (treatment_variable / propensity_score).sum()
        ATE_second_part = (dependent_variable * (1 - treatment_variable) / (1 - propensity_score)).sum() / ((1 - treatment_variable) / (1 - propensity_score)).sum()
        return ATE_first_part - ATE_second_part
    else:
        ATT_first_part = (treatment_variable / treatment_variable.mean() * dependent_variable).mean()
        ATT_second_part_top = ((1 - treatment_variable) * propensity_score / (1 - propensity_score) * dependent_variable).mean()
        ATT_second_part_down = ((1 - treatment_variable) * propensity_score / (1 - propensity_score)).mean()
        return ATT_first_part - ATT_second_part_top / ATT_second_part_down


def propensity_score_regression(dependent_variable, treatment_variable, propensity_score, cov_type=None, target_type="final_model", output_tables=False):
    """
    Propensity Score Regression (Outcome Regression / Regression Adjustment) to estimate ATE.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Binary treatment variable.
        propensity_score (pd.Series): Propensity scores.
        cov_type (str or None): Covariance estimator. Use "HC1" for robust.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "propensity_score_regression")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)

    if cov_type is None:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis=1))).fit()
    else:
        OLS_model = sm.OLS(dependent_variable, sm.add_constant(pd.concat([treatment_variable, propensity_score], axis=1))).fit(cov_type=cov_type)

    if output_tables:
        print(OLS_model.summary())

    if target_type == "neg_pvalue":
        return -OLS_model.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return OLS_model.rsquared_adj
    elif target_type == "final_model":
        return OLS_model


def propensity_score_double_robust_estimator_augmented_IPW(dependent_variable, treatment_variable, propensity_score, covariate_variables, cov_type=None):
    """
    Double Robust Estimator (Augmented IPW / AIPW) to estimate ATE.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Binary treatment variable.
        propensity_score (pd.Series): Propensity scores.
        covariate_variables (pd.DataFrame): Covariate variables.
        cov_type (str or None): Covariance estimator. Use "HC1" for robust.
    """
    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis=1).astype(float)
    if cov_type is None:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit()
    else:
        regression = sm.OLS(dependent_variable, sm.add_constant(X)).fit(cov_type=cov_type)

    IPW = treatment_variable / propensity_score + (1 - treatment_variable) / (1 - propensity_score)

    selected_X = X.loc[treatment_variable[treatment_variable == 0].index].copy()
    selected_X[treatment_variable.name] = 1
    control_group_constructed_output = pd.Series(regression.predict(sm.add_constant(selected_X, has_constant="add")), index=selected_X.index)
    treatment_group_output = dependent_variable.loc[treatment_variable[treatment_variable == 1].index]

    treatment_weighted = (treatment_group_output * IPW.loc[treatment_group_output.index] / IPW.loc[treatment_group_output.index].sum()).sum()
    control_weighted = (control_group_constructed_output * IPW.loc[control_group_constructed_output.index] / IPW.loc[control_group_constructed_output.index].sum()).sum()
    ATE = treatment_weighted - control_weighted
    return ATE


def propensity_score_double_robust_estimator_IPW_regression_adjustment(dependent_variable, treatment_variable, covariate_variables, propensity_score, cov_type=None, target_type="final_model", output_tables=False):
    """
    Double Robust Estimator (IPW Regression Adjustment / IPW-RA) to estimate ATE.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Binary treatment variable.
        covariate_variables (pd.DataFrame): Covariate variables.
        propensity_score (pd.Series): Propensity scores.
        cov_type (str or None): Covariance estimator. Use "HC1" for robust.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "propensity_score_double_robust_estimator_IPW_regression_adjustment")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    IPW = treatment_variable / propensity_score + (1 - treatment_variable) / (1 - propensity_score)
    IPW = IPW ** 0.5

    if covariate_variables is None:
        X = treatment_variable
    else:
        X = pd.concat([treatment_variable, covariate_variables], axis=1).astype(float)
    if cov_type is None:
        regression = sm.WLS(dependent_variable, sm.add_constant(X), weights=IPW).fit()
    else:
        regression = sm.WLS(dependent_variable, sm.add_constant(X), weights=IPW).fit(cov_type=cov_type)

    if output_tables:
        print(regression.summary())

    if target_type == "neg_pvalue":
        return -regression.pvalues[treatment_variable.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression


def IV_2SLS_regression(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_info="nonrobust", target_type="final_model", output_tables=False):
    """
    Instrument Variable Two-Stage Least Squares (IV-2SLS) regression.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Target treatment variable.
        IV_variable (pd.Series or pd.DataFrame): Instrument variable(s).
        covariate_variables (pd.DataFrame or None): Covariate variables.
        cov_info (str or dict): Covariance estimator.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if isinstance(cov_info, str) and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported!")
    elif isinstance(cov_info, dict) and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "IV_2SLS_regression")

    dependent_variable = dependent_variable.astype(float)
    treatment_variable = treatment_variable.astype(float)
    IV_variable = IV_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    # First stage
    if covariate_variables is None:
        first_step_X = IV_variable
    else:
        first_step_X = pd.concat([IV_variable, covariate_variables], axis=1).astype(float)
    if isinstance(cov_info, str):
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        first_step_regression = sm.OLS(treatment_variable, sm.add_constant(first_step_X)).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})
    predicted_treatment_result = pd.Series(first_step_regression.predict(sm.add_constant(first_step_X)), index=treatment_variable.index)
    predicted_treatment_result.name = treatment_variable.name

    # Second stage
    if covariate_variables is None:
        second_step_X = predicted_treatment_result
    else:
        second_step_X = pd.concat([predicted_treatment_result, covariate_variables], axis=1).astype(float)
    if isinstance(cov_info, str):
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        second_step_regression = sm.OLS(dependent_variable, sm.add_constant(second_step_X)).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})

    if output_tables:
        print(second_step_regression.summary())

    if target_type == "neg_pvalue":
        return -second_step_regression.pvalues[predicted_treatment_result.name]
    elif target_type == "rsquared":
        return second_step_regression.rsquared_adj
    elif target_type == "final_model":
        return second_step_regression


def IV_2SLS_IV_setting_test(dependent_variable, treatment_variable, IV_variable, covariate_variables, cov_type=None):
    """
    Run IV diagnostics for relevance and falsification checks.

    Exclusion restriction is not directly testable in a just-identified IV
    design. The returned residual_falsification_check is only a heuristic:
    it regresses residuals from Y on T and X against Z conditional on X.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        treatment_variable (pd.Series): Target treatment variable.
        IV_variable (pd.Series or pd.DataFrame): Instrument variable(s).
        covariate_variables (pd.DataFrame or None): Covariate variables.
        cov_type (str or None): Covariance estimator.

    Returns:
        dict: Diagnostic results with first_stage, reduced_form, and residual_falsification_check.
    """
    IV_variable = _as_dataframe(IV_variable, "instrument").astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if covariate_variables is None:
        first_stage_X = IV_variable
    else:
        first_stage_X = pd.concat([IV_variable, covariate_variables], axis=1).astype(float)

    if cov_type is None:
        first_stage_OLS = sm.OLS(treatment_variable, sm.add_constant(first_stage_X)).fit()
    else:
        first_stage_OLS = sm.OLS(treatment_variable, sm.add_constant(first_stage_X)).fit(cov_type=cov_type)

    instrument_names = list(IV_variable.columns)
    param_names = list(first_stage_OLS.params.index)
    restriction_matrix = np.zeros((len(instrument_names), len(param_names)))
    for row_index, instrument_name in enumerate(instrument_names):
        restriction_matrix[row_index, param_names.index(instrument_name)] = 1
    first_stage_ftest = first_stage_OLS.f_test(restriction_matrix)

    if cov_type is None:
        reduced_form_OLS = sm.OLS(dependent_variable, sm.add_constant(first_stage_X)).fit()
    else:
        reduced_form_OLS = sm.OLS(dependent_variable, sm.add_constant(first_stage_X)).fit(cov_type=cov_type)

    if covariate_variables is None:
        restriction_test_X = treatment_variable
    else:
        restriction_test_X = pd.concat([treatment_variable, covariate_variables], axis=1).astype(float)
    if cov_type is None:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit()
    else:
        restriction_test_OLS = sm.OLS(dependent_variable, sm.add_constant(restriction_test_X)).fit(cov_type=cov_type)
    residual_series = pd.Series(restriction_test_OLS.resid, index=restriction_test_X.index)
    if cov_type is None:
        residual_falsification_OLS = sm.OLS(residual_series, sm.add_constant(first_stage_X)).fit()
    else:
        residual_falsification_OLS = sm.OLS(residual_series, sm.add_constant(first_stage_X)).fit(cov_type=cov_type)

    return {
        "first_stage": first_stage_OLS,
        "reduced_form": reduced_form_OLS,
        "first_stage_partial_f": _f_test_scalar(first_stage_ftest.fvalue),
        "first_stage_partial_f_pvalue": _f_test_scalar(first_stage_ftest.pvalue),
        "residual_falsification_check": residual_falsification_OLS,
        # Backward-compatible alias. Prefer "first_stage" in new code.
        "relevant_condition": first_stage_OLS,
    }


def Static_Diff_in_Diff_regression(dependent_variable, treatment_entity_dummy, treatment_finished_dummy, covariate_variables, entity_effect=False, time_effect=False, other_effect=None, cov_type="unadjusted", target_type="final_model", output_tables=False):
    """
    Static Difference-in-Differences regression for panel data.

    Args:
        dependent_variable (pd.Series): Target dependent variable with entity-time multi-index.
        treatment_entity_dummy (pd.Series): Treatment group dummy with entity-time multi-index.
        treatment_finished_dummy (pd.Series): Post-treatment dummy with entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        entity_effect (bool): Include entity fixed effects.
        time_effect (bool): Include time fixed effects.
        other_effect (pd.DataFrame or None): Other fixed effects.
        cov_type (str): "unadjusted", "robust", "cluster_entity", "cluster_time", or "cluster_both".
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "Static_Diff_in_Diff_regression")
    count_effects = 0
    if entity_effect:
        count_effects += 1
    if time_effect:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed!")

    dependent_variable = dependent_variable.astype(float)
    treatment_entity_dummy = treatment_entity_dummy.astype(float)
    treatment_finished_dummy = treatment_finished_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if list(treatment_entity_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_entity_dummy Input Error!")
    if list(treatment_finished_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("treatment_finished_dummy Input Error!")

    treatment_entity_dummy.name = "treatment_group"
    treatment_finished_dummy.name = "treated"
    beta = treatment_entity_dummy * treatment_finished_dummy
    beta.name = "treatment_group_treated"
    if covariate_variables is None:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy], axis=1)
    else:
        X = pd.concat([beta, treatment_entity_dummy, treatment_finished_dummy, covariate_variables], axis=1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)

    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type=cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_time=True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    if output_tables:
        print(regression)

    if target_type == "neg_pvalue":
        return -regression.pvalues[beta.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression


def Staggered_Diff_in_Diff_regression(dependent_variable, entity_treatment_dummy, covariate_variables, entity_effect=True, time_effect=True, other_effect=None, cov_type="unadjusted", target_type="final_model", output_tables=False):
    """
    Staggered Difference-in-Differences regression for panel data.

    Args:
        dependent_variable (pd.Series): Target dependent variable with entity-time multi-index.
        entity_treatment_dummy (pd.Series): Treatment dummy with entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        entity_effect (bool): Include entity fixed effects.
        time_effect (bool): Include time fixed effects.
        other_effect (pd.DataFrame or None): Other fixed effects.
        cov_type (str): "unadjusted", "robust", "cluster_entity", "cluster_time", or "cluster_both".
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "Staggered_Diff_in_Diff_regression")
    count_effects = 0
    if entity_effect:
        count_effects += 1
    if time_effect:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed!")

    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error!")

    entity_treatment_dummy.name = "treatment_entity_treated"
    if covariate_variables is None:
        X = entity_treatment_dummy
    else:
        X = pd.concat([entity_treatment_dummy, covariate_variables], axis=1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)

    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type=cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_time=True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    if output_tables:
        print(regression)

    if target_type == "neg_pvalue":
        return -regression.pvalues[entity_treatment_dummy.name]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression


def Staggered_Diff_in_Diff_Event_Study_regression(dependent_variable, entity_treatment_dummy, covariate_variables, see_back_length: int = 4, see_forward_length: int = 3, entity_effect=True, time_effect=True, other_effect=None, cov_type="unadjusted", target_type="final_model", output_tables=False):
    """
    Staggered DID Event Study regression for panel data.

    Args:
        dependent_variable (pd.Series): Target dependent variable with entity-time multi-index.
        entity_treatment_dummy (pd.Series): Treatment dummy with entity-time multi-index.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        see_back_length (int): Length of pre-treatment observation window.
        see_forward_length (int): Length of post-treatment observation window.
        entity_effect (bool): Include entity fixed effects.
        time_effect (bool): Include time fixed effects.
        other_effect (pd.DataFrame or None): Other fixed effects.
        cov_type (str): "unadjusted", "robust", "cluster_entity", "cluster_time", or "cluster_both".
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if cov_type not in ["unadjusted", "robust", "cluster_entity", "cluster_time", "cluster_both"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "Staggered_Diff_in_Diff_Event_Study_regression")
    count_effects = 0
    if entity_effect:
        count_effects += 1
    if time_effect:
        count_effects += 1
    if other_effect is not None:
        count_effects += other_effect.shape[1]
    if count_effects > 2:
        raise RuntimeError("At most two effects allowed!")

    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if list(entity_treatment_dummy.map(int).sort_values().unique()) != [0, 1]:
        raise RuntimeError("entity_treatment_dummy Input Error!")

    # Construct event study variables
    entity_index_name, time_index_name = entity_treatment_dummy.index.names[0], entity_treatment_dummy.index.names[1]
    treatment_name = entity_treatment_dummy.name
    data_df = entity_treatment_dummy.reset_index()
    all_entity_list = list(data_df[entity_index_name].unique())
    all_time_list = list(data_df[time_index_name].unique())
    all_time_list.sort()

    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        temp_df = temp_df.sort_values(by=time_index_name)
        if temp_df[temp_df[treatment_name] == 0].shape[0] == 0:
            raise RuntimeError(f"Entity {each_entity} was keeping implementing the policy since the start!")

    if see_back_length < 4 or see_forward_length < 3:
        raise RuntimeError("See back/forward length too few!")
    elif see_back_length + see_forward_length >= len(all_time_list):
        raise RuntimeError("See back/forward length too large!")

    # Construct Lead-Lag Dummy Variables (set Lead_D1 as default)
    Lead_column_name_list = ["Lead_D" + str(see_back_length) + "+"]
    for i in np.arange(see_back_length - 1, 1, -1):
        Lead_column_name_list.append("Lead_D" + str(i))
    Lag_column_name_list = []
    for i in np.arange(1, see_forward_length, 1):
        Lag_column_name_list.append("Lag_D" + str(i))
    Lag_column_name_list.append("Lag_D" + str(see_forward_length) + "+")
    Lead_and_Lag_column_name_list = Lead_column_name_list + ["D0"] + Lag_column_name_list

    considered_data_df = data_df[[entity_index_name, time_index_name]]
    considered_data_df[Lead_and_Lag_column_name_list] = np.nan
    for each_entity in all_entity_list:
        temp_df = data_df[data_df[entity_index_name] == each_entity]
        check_series = temp_df[treatment_name] - temp_df[treatment_name].shift().fillna(0)
        if check_series[check_series == 1].shape[0] == 0:
            considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list] = 0
            continue
        policy_time_index = check_series[check_series == 1].index[0]
        for each_index in temp_df.index:
            corresponding_each_time = temp_df.loc[each_index, time_index_name]
            if each_index - policy_time_index <= -see_back_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lead_D" + str(see_back_length) + "+"] = 1
            elif each_index - policy_time_index > -see_back_length and each_index - policy_time_index < -1:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lead_D" + str(policy_time_index - each_index)] = 1
            elif each_index == policy_time_index:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "D0"] = 1
            elif each_index - policy_time_index > 0 and each_index - policy_time_index < see_forward_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lag_D" + str(each_index - policy_time_index)] = 1
            elif each_index - policy_time_index >= see_forward_length:
                considered_data_df.loc[(considered_data_df[entity_index_name] == each_entity) & (considered_data_df[time_index_name] == corresponding_each_time), "Lag_D" + str(see_forward_length) + "+"] = 1
        considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list] = considered_data_df.loc[considered_data_df[entity_index_name] == each_entity, Lead_and_Lag_column_name_list].fillna(0)
    considered_data_df = considered_data_df.set_index([entity_index_name, time_index_name])

    if covariate_variables is None:
        X = considered_data_df
    else:
        X = pd.concat([considered_data_df, covariate_variables], axis=1).astype(float)
    if count_effects == 0:
        X = sm.add_constant(X)

    if cov_type in ["unadjusted", "robust"]:
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type=cov_type)
    elif cov_type == "cluster_entity":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    elif cov_type == "cluster_time":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_time=True)
    elif cov_type == "cluster_both":
        regression = PanelOLS(dependent_variable, X, entity_effects=entity_effect, time_effects=time_effect, other_effects=other_effect, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    if output_tables:
        print(regression)

    if target_type == "neg_pvalue":
        return -regression.pvalues["D0"]
    elif target_type == "rsquared":
        return regression.rsquared
    elif target_type == "final_model":
        return regression


def Staggered_Diff_in_Diff_Event_Study_visualization(regression_model, see_back_length: int = 4, see_forward_length: int = 3):
    """
    Visualize Staggered DID Event Study results.

    Args:
        regression_model: Regression result from Staggered_Diff_in_Diff_Event_Study_regression.
        see_back_length (int): Pre-treatment observation window length.
        see_forward_length (int): Post-treatment observation window length.

    Returns:
        matplotlib.figure.Figure: The event study plot.
    """
    Lead_column_name_list = ["Lead_D" + str(see_back_length) + "+"]
    for i in np.arange(see_back_length - 1, 1, -1):
        Lead_column_name_list.append("Lead_D" + str(i))
    Lag_column_name_list = []
    for i in np.arange(1, see_forward_length, 1):
        Lag_column_name_list.append("Lag_D" + str(i))
    Lag_column_name_list.append("Lag_D" + str(see_forward_length) + "+")
    Lead_and_Lag_column_name_list = Lead_column_name_list + ["D0"] + Lag_column_name_list

    available_terms = [name for name in Lead_and_Lag_column_name_list if name in regression_model.params.index]
    x_positions = np.arange(len(available_terms))
    estimates = regression_model.params.loc[available_terms].astype(float)
    intervals = regression_model.conf_int().loc[available_terms]

    fig, ax = plt.subplots()
    ax.plot(x_positions, estimates, marker="o")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(available_terms, rotation=45)
    ax.set_ylabel("Estimated Coefficients")
    ax.set_title("Event Study")
    ax.axhline(y=0, color="g", linestyle="--")
    if "D0" in available_terms:
        ax.axvline(x=available_terms.index("D0"), color="g", linestyle="--")
    lower = intervals.iloc[:, 0].astype(float)
    upper = intervals.iloc[:, 1].astype(float)
    ax.vlines(x_positions, lower, upper, color="#f44336")
    ax.plot(x_positions, lower, "_", color="#f44336")
    ax.plot(x_positions, upper, "_", color="#f44336")
    fig.tight_layout()
    return fig


def Sharp_Regression_Discontinuity_Design_regression(dependent_variable, entity_treatment_dummy, running_variable, covariate_variables, running_variable_cutoff, running_variable_bandwidth, kernel_choice="uniform", cov_info="nonrobust", target_type="final_model", output_tables=False):
    """
    Sharp RDD Local Linear Regression.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        entity_treatment_dummy (pd.Series): Treatment dummy.
        running_variable (pd.Series): Running variable.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        running_variable_cutoff (float): Cutoff threshold.
        running_variable_bandwidth (float or None): Bandwidth. None for full sample.
        kernel_choice (str): "uniform", "triangle", or "Epanechnikov".
        cov_info (str or dict): Covariance estimator.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice unsupported!")
    if isinstance(cov_info, str) and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported!")
    elif isinstance(cov_info, dict) and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "Sharp_Regression_Discontinuity_Design_regression")
    if running_variable_bandwidth is not None and running_variable_bandwidth <= 0:
        raise RuntimeError("Bandwidth must be larger than 0!")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of range!")

    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if running_variable_bandwidth is None:
        running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    selected_running_variable = running_variable[(running_variable >= running_variable_cutoff - running_variable_bandwidth) & (running_variable <= running_variable_cutoff + running_variable_bandwidth)]
    dependent_variable = dependent_variable.loc[selected_running_variable.index]
    entity_treatment_dummy = entity_treatment_dummy.loc[selected_running_variable.index]
    if covariate_variables is not None:
        covariate_variables = covariate_variables.loc[selected_running_variable.index].astype(float)
    if isinstance(cov_info, dict) and list(cov_info.keys())[0] == "cluster":
        cov_info["cluster"] = cov_info["cluster"].loc[selected_running_variable.index]
    demeaned_selected_running_variable = selected_running_variable - running_variable_cutoff
    demeaned_selected_running_variable.name = "demeaned_" + selected_running_variable.name
    demeaned_selected_running_interaction_variable = demeaned_selected_running_variable * entity_treatment_dummy
    demeaned_selected_running_interaction_variable.name = "demeaned_interaction_" + entity_treatment_dummy.name

    if kernel_choice == "uniform":
        weight = pd.Series(index=selected_running_variable.index).fillna(1 / selected_running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight = 1 - ((selected_running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = selected_running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    if covariate_variables is not None:
        regression_formula = dependent_variable.name + " ~ " + entity_treatment_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable, covariate_variables], axis=1)
    else:
        regression_formula = dependent_variable.name + " ~ " + entity_treatment_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable], axis=1)

    if isinstance(cov_info, str):
        model = smf.wls(regression_formula, complete_dataset, weights=weight).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model = smf.wls(regression_formula, complete_dataset, weights=weight).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model = smf.wls(regression_formula, complete_dataset, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})

    if output_tables:
        print(model.summary())

    if target_type == "neg_pvalue":
        return -model.pvalues[entity_treatment_dummy.name]
    elif target_type == "rsquared":
        return model.rsquared
    elif target_type == "final_model":
        return model


def Fuzzy_Regression_Discontinuity_Design_regression(dependent_variable, entity_treatment_dummy, running_variable, covariate_variables, running_variable_cutoff, running_variable_bandwidth, kernel_choice="uniform", cov_info="nonrobust", target_type="summary", output_tables=False):
    """
    Two-step Fuzzy RDD Local Linear Regression.

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        entity_treatment_dummy (pd.Series): Treatment dummy.
        running_variable (pd.Series): Running variable.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        running_variable_cutoff (float): Cutoff threshold.
        running_variable_bandwidth (float or None): Bandwidth. None for full sample.
        kernel_choice (str): "uniform", "triangle", or "Epanechnikov".
        cov_info (str or dict): Covariance estimator.
        target_type (str): "summary", "estimator", or "final_models".
        output_tables (bool): Whether to print regression tables.
    """
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice unsupported!")
    if isinstance(cov_info, str) and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported!")
    elif isinstance(cov_info, dict) and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("summary", "estimator", "final_models"), "Fuzzy_Regression_Discontinuity_Design_regression")
    if running_variable_bandwidth is not None and running_variable_bandwidth <= 0:
        raise RuntimeError("Bandwidth must be larger than 0!")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of range!")

    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    if running_variable_bandwidth is None:
        running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    selected_running_variable = running_variable[(running_variable >= running_variable_cutoff - running_variable_bandwidth) & (running_variable <= running_variable_cutoff + running_variable_bandwidth)]
    dependent_variable = dependent_variable.loc[selected_running_variable.index]
    entity_treatment_dummy = entity_treatment_dummy.loc[selected_running_variable.index]
    if covariate_variables is not None:
        covariate_variables = covariate_variables.loc[selected_running_variable.index].astype(float)
    if isinstance(cov_info, dict) and list(cov_info.keys())[0] == "cluster":
        cov_info["cluster"] = cov_info["cluster"].loc[selected_running_variable.index]
    should_be_treated_dummy = selected_running_variable.map(lambda x: 1 if x >= running_variable_cutoff else 0)
    should_be_treated_dummy.name = selected_running_variable.name + "_dummy"
    demeaned_selected_running_variable = selected_running_variable - running_variable_cutoff
    demeaned_selected_running_variable.name = "demeaned_" + selected_running_variable.name
    demeaned_selected_running_interaction_variable = demeaned_selected_running_variable * should_be_treated_dummy
    demeaned_selected_running_interaction_variable.name = "demeaned_interaction_" + selected_running_variable.name

    if kernel_choice == "uniform":
        weight = pd.Series(index=selected_running_variable.index).fillna(1 / selected_running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight = 1 - ((selected_running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = selected_running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    if covariate_variables is not None:
        regression_formula_1 = dependent_variable.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        regression_formula_2 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, should_be_treated_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable, covariate_variables], axis=1)
    else:
        regression_formula_1 = dependent_variable.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        regression_formula_2 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + demeaned_selected_running_variable.name + " + " + demeaned_selected_running_interaction_variable.name
        complete_dataset = pd.concat([dependent_variable, entity_treatment_dummy, should_be_treated_dummy, demeaned_selected_running_variable, demeaned_selected_running_interaction_variable], axis=1)

    if isinstance(cov_info, str):
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights=weight).fit(cov_type=cov_info)
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights=weight).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights=weight).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights=weight).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_1 = smf.wls(regression_formula_1, complete_dataset, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})
        model_2 = smf.wls(regression_formula_2, complete_dataset, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})

    if output_tables:
        print(model_1.summary())
        print(model_2.summary())

    reduced_form_jump = float(model_1.params[should_be_treated_dummy.name])
    first_stage_jump = float(model_2.params[should_be_treated_dummy.name])
    wald_late = reduced_form_jump / first_stage_jump if first_stage_jump != 0 else np.nan

    reduced_form_se = float(model_1.bse[should_be_treated_dummy.name])
    first_stage_se = float(model_2.bse[should_be_treated_dummy.name])
    if first_stage_jump == 0:
        approx_delta_se = np.inf
    else:
        # Approximate delta method that ignores covariance between stages.
        approx_delta_se = float(np.sqrt((reduced_form_se / first_stage_jump) ** 2 + ((reduced_form_jump * first_stage_se) / (first_stage_jump ** 2)) ** 2))

    if np.isfinite(approx_delta_se):
        ci_low = float(wald_late - 1.96 * approx_delta_se)
        ci_high = float(wald_late + 1.96 * approx_delta_se)
    else:
        ci_low = np.nan
        ci_high = np.nan

    summary = {
        "wald_late": float(wald_late),
        "reduced_form_jump": reduced_form_jump,
        "first_stage_jump": first_stage_jump,
        "reduced_form_se": reduced_form_se,
        "first_stage_se": first_stage_se,
        "first_stage_pvalue": float(model_2.pvalues[should_be_treated_dummy.name]),
        "approx_delta_se": approx_delta_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "within_bandwidth_n": int(selected_running_variable.shape[0]),
        "bandwidth": float(running_variable_bandwidth),
        "weak_first_stage": bool(abs(first_stage_jump) < 0.05 or model_2.pvalues[should_be_treated_dummy.name] > 0.05),
        "models": [model_1, model_2],
        "se_note": "Approximate delta-method SE ignores covariance between reduced-form and first-stage estimates; bootstrap or dedicated IV/RDD software is preferred for serious inference.",
    }

    if target_type == "summary":
        return summary
    elif target_type == "estimator":
        return wald_late
    elif target_type == "final_models":
        return [model_1, model_2]


def Fuzzy_RDD_Global_Polynomial_Estimator_regression(dependent_variable, entity_treatment_dummy, running_variable, covariate_variables, running_variable_cutoff, max_order, kernel_choice="uniform", cov_info="nonrobust", target_type="final_model", output_tables=False):
    """
    Fuzzy RDD Global Polynomial Estimator (two-step).

    Args:
        dependent_variable (pd.Series): Target dependent variable.
        entity_treatment_dummy (pd.Series): Treatment dummy.
        running_variable (pd.Series): Running variable.
        covariate_variables (pd.DataFrame or None): Covariate variables.
        running_variable_cutoff (float): Cutoff threshold.
        max_order (int): Highest polynomial order (>= 1).
        kernel_choice (str): "uniform", "triangle", or "Epanechnikov".
        cov_info (str or dict): Covariance estimator.
        target_type (str): "neg_pvalue", "rsquared", or "final_model".
        output_tables (bool): Whether to print regression tables.
    """
    if kernel_choice not in ["uniform", "triangle", "Epanechnikov"]:
        raise RuntimeError("Kernel function choice unsupported!")
    if isinstance(cov_info, str) and cov_info not in ["nonrobust", "HC0", "HC1", "HC2", "HC3"]:
        raise RuntimeError("Covariance type input unsupported!")
    elif isinstance(cov_info, dict) and list(cov_info.keys())[0] not in ["HAC", "cluster"]:
        raise RuntimeError("Covariance type input unsupported!")
    _validate_target_type(target_type, ("neg_pvalue", "rsquared", "final_model"), "Fuzzy_RDD_Global_Polynomial_Estimator_regression")
    if running_variable[running_variable > running_variable_cutoff].shape[0] == 0 or running_variable[running_variable < running_variable_cutoff].shape[0] == 0:
        raise RuntimeError("Running variable cutoff is out of range!")
    if max_order < 1:
        raise RuntimeError("max_order must be >= 1!")

    dependent_variable = dependent_variable.astype(float)
    entity_treatment_dummy = entity_treatment_dummy.astype(float)
    running_variable = running_variable.astype(float)
    if covariate_variables is not None:
        covariate_variables = covariate_variables.astype(float)

    should_be_treated_dummy = running_variable.map(lambda x: 1 if x >= running_variable_cutoff else 0)
    should_be_treated_dummy.name = running_variable.name + "_dummy"

    running_variable_bandwidth = max(running_variable.max() - running_variable_cutoff, running_variable_cutoff - running_variable.min())
    if kernel_choice == "uniform":
        weight = pd.Series(index=running_variable.index).fillna(1 / running_variable.shape[0])
    elif kernel_choice == "triangle":
        weight = 1 - ((running_variable - running_variable_cutoff) / running_variable_bandwidth).abs()
    elif kernel_choice == "Epanechnikov":
        weight = running_variable.map(lambda x: 0.75 * (1 - np.abs(((x - running_variable_cutoff) / running_variable_bandwidth)) ** 2))

    # Step 1: construct polynomial terms and run first stage
    all_constructed_terms_list_step_1 = []
    for each_order in range(1, max_order + 1):
        demeaned_running_variable = (running_variable - running_variable_cutoff) ** each_order
        demeaned_running_variable.name = "demeaned_order_" + str(each_order) + "_" + running_variable.name
        demeaned_running_interaction_variable = demeaned_running_variable * should_be_treated_dummy
        demeaned_running_interaction_variable.name = "demeaned_order_" + str(each_order) + "_interaction_" + running_variable.name
        all_constructed_terms_list_step_1.append(demeaned_running_variable)
        all_constructed_terms_list_step_1.append(demeaned_running_interaction_variable)
    all_constructed_terms_list_step_1 = pd.concat(all_constructed_terms_list_step_1, axis=1)
    all_constructed_terms_name_list_step_1 = list(all_constructed_terms_list_step_1.columns)

    if covariate_variables is not None:
        regression_formula_1 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + " + ".join(all_constructed_terms_name_list_step_1) + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset_1 = pd.concat([entity_treatment_dummy, should_be_treated_dummy, all_constructed_terms_list_step_1, covariate_variables.astype(float)], axis=1)
    else:
        regression_formula_1 = entity_treatment_dummy.name + " ~ " + should_be_treated_dummy.name + " + " + " + ".join(all_constructed_terms_name_list_step_1)
        complete_dataset_1 = pd.concat([entity_treatment_dummy, should_be_treated_dummy, all_constructed_terms_list_step_1], axis=1)

    if isinstance(cov_info, str):
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights=weight).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights=weight).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_1 = smf.wls(regression_formula_1, complete_dataset_1, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})
    entity_treatment_dummy_hat = pd.Series(model_1.predict(complete_dataset_1[complete_dataset_1.columns[1:]]))
    entity_treatment_dummy_hat.name = entity_treatment_dummy.name

    # Step 2: construct polynomial terms and run second stage
    all_constructed_terms_list_step_2 = []
    for each_order in range(1, max_order + 1):
        demeaned_running_variable = (running_variable - running_variable_cutoff) ** each_order
        demeaned_running_variable.name = "demeaned_order_" + str(each_order) + "_" + running_variable.name
        demeaned_running_interaction_variable = demeaned_running_variable * entity_treatment_dummy_hat
        demeaned_running_interaction_variable.name = "demeaned_order_" + str(each_order) + "_interaction_" + running_variable.name
        all_constructed_terms_list_step_2.append(demeaned_running_variable)
        all_constructed_terms_list_step_2.append(demeaned_running_interaction_variable)
    all_constructed_terms_list_step_2 = pd.concat(all_constructed_terms_list_step_2, axis=1)
    all_constructed_terms_name_list_step_2 = list(all_constructed_terms_list_step_2.columns)

    if covariate_variables is not None:
        regression_formula_2 = dependent_variable.name + " ~ " + entity_treatment_dummy_hat.name + " + " + " + ".join(all_constructed_terms_name_list_step_2) + " + " + " + ".join(list(covariate_variables.columns))
        complete_dataset_2 = pd.concat([dependent_variable, entity_treatment_dummy_hat, all_constructed_terms_list_step_2, covariate_variables.astype(float)], axis=1)
    else:
        regression_formula_2 = dependent_variable.name + " ~ " + entity_treatment_dummy_hat.name + " + " + " + ".join(all_constructed_terms_name_list_step_2)
        complete_dataset_2 = pd.concat([dependent_variable, entity_treatment_dummy_hat, all_constructed_terms_list_step_2], axis=1)

    if isinstance(cov_info, str):
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights=weight).fit(cov_type=cov_info)
    elif list(cov_info.keys())[0] == "HAC":
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights=weight).fit(cov_type="HAC", cov_kwds={"maxlags": cov_info["HAC"]})
    elif list(cov_info.keys())[0] == "cluster":
        model_2 = smf.wls(regression_formula_2, complete_dataset_2, weights=weight).fit(cov_type="cluster", cov_kwds={"groups": cov_info["cluster"]})

    if output_tables:
        print(model_1.summary())
        print(model_2.summary())

    if target_type == "neg_pvalue":
        return -model_2.pvalues[entity_treatment_dummy_hat.name]
    elif target_type == "rsquared":
        return model_2.rsquared
    elif target_type == "final_model":
        return model_2
