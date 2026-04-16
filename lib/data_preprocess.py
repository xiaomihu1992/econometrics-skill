from __future__ import annotations

import json
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
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            column_info["Category"].append(col)
        elif pd.api.types.is_numeric_dtype(series):
            column_info["Numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_info["Datetime"].append(col)
        else:
            column_info["Others"].append(col)

    if len(json.dumps(column_info)) > 2000:
        column_info["Numeric"] = column_info["Numeric"][0:5] + ["Too many cols, omission here..."]
    return column_info
