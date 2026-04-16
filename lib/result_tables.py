from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd


def _get_attr(model: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            return value() if callable(value) and name in {"conf_int"} else value
    return default


def _as_series(value: Any, name: str) -> pd.Series:
    if value is None:
        return pd.Series(dtype=float, name=name)
    if isinstance(value, pd.Series):
        return value.rename(name)
    return pd.Series(value, name=name)


def _format_number(value: Any, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def significance_stars(pvalue: Any) -> str:
    if pvalue is None or pd.isna(pvalue):
        return ""
    pvalue = float(pvalue)
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def extract_model_results(model: Any, terms: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Extract coefficient-level results from statsmodels or linearmodels objects.

    Returns columns: term, estimate, std_error, p_value, ci_low, ci_high.
    """
    params = _as_series(_get_attr(model, ["params"]), "estimate")
    std_errors = _as_series(_get_attr(model, ["bse", "std_errors"]), "std_error")
    pvalues = _as_series(_get_attr(model, ["pvalues"]), "p_value")

    result = pd.concat([params, std_errors, pvalues], axis=1)
    result.index.name = "term"
    result = result.reset_index()

    conf_int = _get_attr(model, ["conf_int"])
    if conf_int is not None:
        ci = pd.DataFrame(conf_int)
        ci.index = params.index
        if ci.shape[1] >= 2:
            ci = ci.iloc[:, :2]
            ci.columns = ["ci_low", "ci_high"]
            result = result.merge(ci.reset_index(names="term"), on="term", how="left")

    if "ci_low" not in result:
        result["ci_low"] = pd.NA
    if "ci_high" not in result:
        result["ci_high"] = pd.NA

    if terms is not None:
        wanted = list(terms)
        result = result[result["term"].isin(wanted)]
        result["term"] = pd.Categorical(result["term"], categories=wanted, ordered=True)
        result = result.sort_values("term")
        result["term"] = result["term"].astype(str)

    return result[["term", "estimate", "std_error", "p_value", "ci_low", "ci_high"]].reset_index(drop=True)


def model_fit_stats(model: Any) -> dict[str, Any]:
    return {
        "N": _get_attr(model, ["nobs"]),
        "R2": _get_attr(model, ["rsquared", "rsquared_overall"]),
        "Adj. R2": _get_attr(model, ["rsquared_adj"]),
    }


def regression_table(
    models: Sequence[Any],
    model_names: Sequence[str] | None = None,
    terms: Iterable[str] | None = None,
    digits: int = 3,
    stars: bool = True,
    include_fit_stats: bool = True,
) -> pd.DataFrame:
    """
    Build a compact regression table for multiple fitted models.

    Coefficients and standard errors are stacked in conventional display form:
    one row for the coefficient, followed by one row with the standard error in
    parentheses. The returned DataFrame can be printed, exported to Markdown via
    `to_markdown_table`, or exported to LaTeX via `df.to_latex()`.
    """
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(models))]
    if len(model_names) != len(models):
        raise ValueError("model_names must have the same length as models.")

    extracted = [extract_model_results(model, terms=terms) for model in models]
    if terms is None:
        ordered_terms: list[str] = []
        for table in extracted:
            for term in table["term"]:
                if term not in ordered_terms:
                    ordered_terms.append(term)
    else:
        ordered_terms = list(terms)

    rows: list[str] = []
    data: dict[str, list[str]] = {name: [] for name in model_names}

    for term in ordered_terms:
        rows.extend([term, f"{term} SE"])
        for name, table in zip(model_names, extracted, strict=True):
            row = table[table["term"] == term]
            if row.empty:
                data[name].extend(["", ""])
                continue
            row = row.iloc[0]
            star_text = significance_stars(row["p_value"]) if stars else ""
            data[name].append(f"{_format_number(row['estimate'], digits)}{star_text}")
            data[name].append(f"({_format_number(row['std_error'], digits)})")

    if include_fit_stats:
        for stat in ["N", "R2", "Adj. R2"]:
            rows.append(stat)
            for name, model in zip(model_names, models, strict=True):
                value = model_fit_stats(model).get(stat)
                if stat == "N" and value is not None and not pd.isna(value):
                    data[name].append(str(int(value)))
                else:
                    data[name].append(_format_number(value, digits))

    table = pd.DataFrame(data, index=rows)
    table.index.name = "term"
    return table


def to_markdown_table(table: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without extra dependencies."""
    rendered = table.reset_index().astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(row[col] for col in headers) + " |")
    return "\n".join(lines)
