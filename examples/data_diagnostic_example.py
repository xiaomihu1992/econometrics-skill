from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SKILL_LIB = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(SKILL_LIB))

from data_preprocess import analyze_dataset, format_dataset_report


def main() -> None:
    df = pd.DataFrame(
        {
            "city id": ["A", "A", "B", "B", "C", "C"],
            "year": [2020, 2021, 2020, 2021, 2020, 2021],
            "treated?": [0, 1, 0, 0, 0, 1],
            "employment_rate": [87.1, 88.4, 82.5, None, 79.2, 95.0],
            "industry": ["service", "service", "manufacturing", "manufacturing", "service", None],
            "median_wage_text": ["5100", "5300", "4700", "4900", "bad_value", "5600"],
            "constant_col": [1, 1, 1, 1, 1, 1],
        }
    )

    analysis = analyze_dataset(
        df,
        role_hints={
            "outcome": "employment_rate",
            "treatment": "treated?",
            "unit": "city id",
            "time": "year",
        },
    )
    print(format_dataset_report(analysis))


if __name__ == "__main__":
    main()
