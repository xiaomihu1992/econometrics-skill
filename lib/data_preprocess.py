from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from typing import Any
from pathlib import Path

import pandas as pd


def load_table(file_path: str | Path, sheet_name: str | int = 0, **kwargs) -> pd.DataFrame:
    """
    Load tabular data from CSV/TSV or Excel files.

    Args:
        file_path: Path to a .csv, .tsv, .xlsx, .xls, or .xlsm file.
        sheet_name: Excel sheet name or index. Ignored for CSV/TSV.
        **kwargs: Extra keyword arguments passed to pandas.

    Returns:
        A pandas DataFrame.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    raise ValueError(f"Unsupported table format: {suffix}. Use CSV, TSV, XLSX, XLS, or XLSM.")


def get_column_info(df: pd.DataFrame) -> dict:
    """
    Analyzes a DataFrame and categorizes its columns based on data types.

    Args:
        df: The DataFrame to be analyzed.

    Returns:
        A dictionary with four keys ('Category', 'Numeric', 'Datetime', 'Others').
        Each key corresponds to a list of column names belonging to that category.
    """
    column_info = {
        "Category": [],
        "Numeric": [],
        "Datetime": [],
        "Others": [],
    }
    for col, series in df.items():
        col_name = str(col)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            column_info["Category"].append(col_name)
        elif pd.api.types.is_numeric_dtype(series):
            column_info["Numeric"].append(col_name)
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_info["Datetime"].append(col_name)
        else:
            column_info["Others"].append(col_name)

    if len(json.dumps(column_info)) > 2000:
        column_info["Numeric"] = column_info["Numeric"][0:5] + ["Too many cols, omission here..."]
    return column_info


def _pct(part: int | float, whole: int | float) -> float:
    if not whole:
        return 0.0
    return round(float(part) / float(whole), 4)


def _clean_column_name(name: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not cleaned:
        cleaned = "unnamed"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def _sample_values(series: pd.Series, limit: int) -> list[str]:
    values = series.dropna().head(limit).tolist()
    return [str(value) for value in values]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    if non_missing.empty:
        return {}
    quantiles = non_missing.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "min": float(non_missing.min()),
        "p01": float(quantiles.loc[0.01]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(non_missing.max()),
        "mean": float(non_missing.mean()),
        "std": float(non_missing.std()) if len(non_missing) > 1 else 0.0,
    }


def _datetime_parse_ratio(series: pd.Series, limit: int = 500) -> float:
    sample = series.dropna().head(limit)
    if sample.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return round(float(parsed.notna().mean()), 4)


def _datetime_summary(series: pd.Series) -> dict[str, str]:
    dt = pd.to_datetime(series, errors="coerce")
    non_missing = dt.dropna()
    if non_missing.empty:
        return {}
    return {
        "min": str(non_missing.min()),
        "max": str(non_missing.max()),
    }


def _looks_like_id(col: str, series: pd.Series) -> bool:
    name = col.lower()
    non_missing = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    if non_missing == 0:
        return False
    name_hint = any(token in name for token in ["id", "code", "编号", "代码", "企业", "城市", "个人", "学校"])
    return name_hint or (non_missing >= 20 and unique / non_missing > 0.9)


def _looks_like_time(col: str, series: pd.Series) -> bool:
    name = col.lower()
    if any(token in name for token in ["year", "month", "date", "time", "period", "年份", "日期", "月份", "时间"]):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    numeric_ratio = pd.to_numeric(series.dropna().head(200), errors="coerce").notna().mean()
    if not pd.isna(numeric_ratio) and numeric_ratio > 0.8:
        return False
    return _datetime_parse_ratio(series, limit=200) > 0.8


def _is_binary_like(series: pd.Series) -> bool:
    values = set(str(value).strip().lower() for value in series.dropna().unique().tolist())
    if not values or len(values) > 2:
        return False
    binary_sets = [
        {"0", "1"},
        {"false", "true"},
        {"no", "yes"},
        {"n", "y"},
        {"control", "treated"},
        {"对照", "处理"},
        {"否", "是"},
    ]
    return values in binary_sets or len(values) == 2


def _looks_like_treatment(col: str, series: pd.Series) -> bool:
    name = col.lower()
    name_hint = any(
        token in name
        for token in [
            "treat",
            "policy",
            "reform",
            "program",
            "eligible",
            "post",
            "did",
            "处理",
            "政策",
            "改革",
            "试点",
        ]
    )
    if name_hint:
        return True
    if not _is_binary_like(series):
        return False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    values = set(str(value).strip().lower() for value in series.dropna().unique().tolist())
    known_treatment_values = {"0", "1", "false", "true", "no", "yes", "n", "y", "control", "treated", "对照", "处理", "否", "是"}
    return values.issubset(known_treatment_values)


def _column_profile(col: str, series: pd.Series, sample_values: int) -> dict[str, Any]:
    missing = int(series.isna().sum())
    non_missing = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    clean_name = _clean_column_name(col)

    profile: dict[str, Any] = {
        "name": str(col),
        "clean_name": clean_name,
        "needs_rename": clean_name != str(col),
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": _pct(missing, len(series)),
        "unique": unique,
        "unique_pct": _pct(unique, non_missing),
        "examples": _sample_values(series, sample_values),
        "flags": [],
    }

    if missing:
        profile["flags"].append("missing_values")
    if unique == 0:
        profile["flags"].append("all_missing")
    elif unique == 1:
        profile["flags"].append("constant")
    if _looks_like_id(col, series):
        profile["flags"].append("id_candidate")
    if _looks_like_time(col, series):
        profile["flags"].append("time_candidate")
    if _is_binary_like(series):
        profile["flags"].append("binary_candidate")
    if _looks_like_treatment(col, series):
        profile["flags"].append("treatment_candidate")

    if pd.api.types.is_numeric_dtype(series):
        profile["kind"] = "numeric"
        profile["summary"] = _numeric_summary(series)
        summary = profile["summary"]
        if summary and summary["p99"] != summary["p01"]:
            iqr = summary["p75"] - summary["p25"]
            if iqr > 0 and (summary["max"] > summary["p75"] + 3 * iqr or summary["min"] < summary["p25"] - 3 * iqr):
                profile["flags"].append("possible_outliers")
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile["kind"] = "datetime"
        profile["summary"] = _datetime_summary(series)
    else:
        profile["kind"] = "categorical_or_text"
        numeric_ratio = pd.to_numeric(series.dropna().head(500), errors="coerce").notna().mean()
        datetime_ratio = _datetime_parse_ratio(series, limit=500) if numeric_ratio <= 0.8 else 0.0
        profile["parse_as_numeric_ratio"] = round(float(numeric_ratio), 4) if not pd.isna(numeric_ratio) else 0.0
        profile["parse_as_datetime_ratio"] = round(float(datetime_ratio), 4) if not pd.isna(datetime_ratio) else 0.0
        if profile["parse_as_numeric_ratio"] > 0.8:
            profile["flags"].append("numeric_stored_as_text")
        if profile["parse_as_datetime_ratio"] > 0.9:
            profile["flags"].append("datetime_stored_as_text")

    return profile


def analyze_dataset(
    data: str | Path | pd.DataFrame,
    sheet_name: str | int = 0,
    role_hints: Mapping[str, str | list[str]] | None = None,
    max_columns: int = 80,
    sample_values: int = 3,
) -> dict[str, Any]:
    """
    Inspect an uploaded tabular dataset and return cleaning/modeling advice.

    Args:
        data: File path or already-loaded DataFrame.
        sheet_name: Excel sheet name or index when data is a path.
        role_hints: Optional mapping such as {"outcome": "Y", "treatment": "T"}.
        max_columns: Maximum number of columns to profile in detail.
        sample_values: Number of example non-missing values per column.

    Returns:
        A JSON-serializable dictionary with overview, column profiles, role
        candidates, issues, and recommended preprocessing actions.
    """
    source = None
    if isinstance(data, (str, Path)):
        source = str(data)
        df = load_table(data, sheet_name=sheet_name)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data must be a file path or pandas DataFrame.")

    role_hints = dict(role_hints or {})
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum()) if n_rows else 0
    profiled_columns = list(df.columns[:max_columns])
    columns = [_column_profile(str(col), df.iloc[:, idx], sample_values) for idx, col in enumerate(profiled_columns)]

    role_candidates = {
        "unit_id": _dedupe([c["name"] for c in columns if "id_candidate" in c["flags"]])[:8],
        "time": _dedupe([c["name"] for c in columns if "time_candidate" in c["flags"]])[:8],
        "treatment": _dedupe([
            c["name"]
            for c in columns
            if "treatment_candidate" in c["flags"] and "time_candidate" not in c["flags"] and "id_candidate" not in c["flags"]
        ])[:8],
        "numeric_outcome_or_control": _dedupe([
            c["name"]
            for c in columns
            if c.get("kind") == "numeric"
            and "id_candidate" not in c["flags"]
            and "time_candidate" not in c["flags"]
            and "treatment_candidate" not in c["flags"]
            and "constant" not in c["flags"]
            and "all_missing" not in c["flags"]
        ])[:12],
        "categorical_control": _dedupe([
            c["name"]
            for c in columns
            if c.get("kind") == "categorical_or_text"
            and "id_candidate" not in c["flags"]
            and "time_candidate" not in c["flags"]
            and "numeric_stored_as_text" not in c["flags"]
            and "datetime_stored_as_text" not in c["flags"]
        ])[:12],
    }

    issues: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    duplicate_columns = [str(col) for col in df.columns[df.columns.duplicated()].tolist()]
    if duplicate_columns:
        issues.append({"severity": "high", "topic": "duplicate_columns", "columns": duplicate_columns[:20]})
        recommendations.append({
            "priority": "high",
            "action": "Rename duplicate columns before selecting variables; duplicate names make formula and column selection ambiguous.",
            "columns": duplicate_columns[:20],
        })

    if duplicate_rows:
        issues.append({"severity": "medium", "topic": "duplicate_rows", "detail": f"{duplicate_rows} duplicate rows detected."})
        recommendations.append({
            "priority": "high",
            "action": "Inspect and remove exact duplicate rows unless they represent valid repeated observations.",
            "columns": [],
        })

    rename_cols = [c["name"] for c in columns if c["needs_rename"]]
    if rename_cols:
        recommendations.append({
            "priority": "high",
            "action": "Rename columns to clean Python identifiers before formula-based OLS/RDD work.",
            "columns": rename_cols[:20],
        })

    high_missing = [c["name"] for c in columns if c["missing_pct"] >= 0.4]
    moderate_missing = [c["name"] for c in columns if 0.05 <= c["missing_pct"] < 0.4]
    if high_missing:
        issues.append({"severity": "high", "topic": "high_missingness", "columns": high_missing[:20]})
        recommendations.append({
            "priority": "high",
            "action": "Do not use high-missingness columns as core variables without a missingness design; consider exclusion, missing indicators, or sensitivity checks.",
            "columns": high_missing[:20],
        })
    if moderate_missing:
        recommendations.append({
            "priority": "medium",
            "action": "Handle missing values explicitly before estimation; compare complete-case results with imputation or missing-indicator variants.",
            "columns": moderate_missing[:20],
        })

    constant_cols = [c["name"] for c in columns if "constant" in c["flags"] or "all_missing" in c["flags"]]
    if constant_cols:
        recommendations.append({
            "priority": "medium",
            "action": "Drop constant or all-missing columns before building covariate matrices.",
            "columns": constant_cols[:20],
        })

    text_numeric = [c["name"] for c in columns if "numeric_stored_as_text" in c["flags"]]
    if text_numeric:
        recommendations.append({
            "priority": "high",
            "action": "Convert numeric-looking text columns with pd.to_numeric(..., errors='coerce') and inspect newly created missing values.",
            "columns": text_numeric[:20],
        })

    text_datetime = [c["name"] for c in columns if "datetime_stored_as_text" in c["flags"]]
    if text_datetime:
        recommendations.append({
            "priority": "medium",
            "action": "Parse date-looking text columns with pd.to_datetime(..., errors='coerce') before panel or timing analysis.",
            "columns": text_datetime[:20],
        })

    outliers = [c["name"] for c in columns if "possible_outliers" in c["flags"]]
    if outliers:
        recommendations.append({
            "priority": "medium",
            "action": "Inspect extreme numeric values; consider winsorization, log transforms, or influence diagnostics rather than silent deletion.",
            "columns": outliers[:20],
        })

    if role_candidates["unit_id"] and role_candidates["time"]:
        recommendations.append({
            "priority": "medium",
            "action": "This looks compatible with panel methods; check duplicate unit-time pairs and sort before DID/event-study analysis.",
            "columns": role_candidates["unit_id"][:3] + role_candidates["time"][:3],
        })

    if role_candidates["treatment"]:
        recommendations.append({
            "priority": "high",
            "action": "Validate treatment coding before estimation; most bundled estimators expect a 0/1 treatment indicator.",
            "columns": role_candidates["treatment"][:8],
        })

    if role_hints:
        recommendations.append({
            "priority": "high",
            "action": "Check user-provided role hints against the column diagnostics before choosing an estimator.",
            "columns": list(role_hints.values()) if all(isinstance(v, str) for v in role_hints.values()) else [],
        })

    return {
        "source": source,
        "overview": {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "profiled_columns": len(profiled_columns),
            "duplicate_rows": duplicate_rows,
            "duplicate_columns": len(duplicate_columns),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1_000_000, 3),
        },
        "column_info": get_column_info(df),
        "columns": columns,
        "role_candidates": role_candidates,
        "issues": issues,
        "recommendations": recommendations,
        "next_questions": [
            "Which column is the outcome Y?",
            "Which column is the treatment T?",
            "What is the unit of observation?",
            "Is there a time variable or policy timing variable?",
            "Which variables were measured before treatment and can be used as controls?",
        ],
    }


def format_dataset_report(analysis: Mapping[str, Any]) -> str:
    """Render analyze_dataset output as a concise Markdown report."""
    overview = analysis["overview"]
    lines = [
        "# Dataset Diagnostic Report",
        "",
        "## Overview",
        f"- Rows: {overview['rows']}",
        f"- Columns: {overview['columns']} ({overview['profiled_columns']} profiled)",
        f"- Duplicate rows: {overview['duplicate_rows']}",
        f"- Duplicate columns: {overview.get('duplicate_columns', 0)}",
        f"- Memory: {overview['memory_mb']} MB",
        "",
        "## Role Candidates",
    ]

    for role, cols in analysis["role_candidates"].items():
        shown = ", ".join(cols) if cols else "None detected"
        lines.append(f"- {role}: {shown}")

    lines.extend(["", "## Recommended Processing"])
    recommendations = analysis.get("recommendations", [])
    if not recommendations:
        lines.append("- No major preprocessing issues detected from automatic checks.")
    else:
        for item in recommendations:
            cols = item.get("columns") or []
            suffix = f" Columns: {', '.join(cols)}." if cols else ""
            lines.append(f"- [{item['priority']}] {item['action']}{suffix}")

    lines.extend(["", "## Questions Before Estimation"])
    for question in analysis.get("next_questions", []):
        lines.append(f"- {question}")

    return "\n".join(lines)
