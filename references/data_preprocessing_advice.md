# Data Preprocessing Advice

Use this reference whenever the user uploads a dataset or gives a table path. Start with automatic diagnostics before choosing an estimator.

## Automatic diagnostic workflow

```python
import sys
from pathlib import Path

skill_dir = Path("/path/to/econometrics-skill")
sys.path.insert(0, str(skill_dir / "lib"))

from data_preprocess import analyze_dataset, format_dataset_report

analysis = analyze_dataset("/path/to/data.xlsx", sheet_name=0)
print(format_dataset_report(analysis))
```

If the user already named key variables, pass them as role hints:

```python
analysis = analyze_dataset(
    "/path/to/data.csv",
    role_hints={
        "outcome": "employment_rate",
        "treatment": "min_wage_reform",
        "unit": "city_id",
        "time": "year",
    },
)
```

## How to interpret the output

- `overview`: row/column counts, duplicate rows, profiled columns, and memory use.
- `column_info`: broad dtype grouping from `get_column_info()`.
- `columns`: per-column profile with missingness, uniqueness, examples, flags, and summaries.
- `role_candidates`: possible unit id, time, treatment, outcome/control, and categorical-control columns.
- `issues`: high-level risks that need attention.
- `recommendations`: prioritized cleaning actions.
- `next_questions`: questions to ask before estimation.

## Recommended response shape

When reporting diagnostics to the user, keep the output practical:

1. Give a one-paragraph dataset summary.
2. List the most important cleaning actions first.
3. Identify likely outcome, treatment, unit, time, and control candidates, but do not pretend automatic detection is certain.
4. Ask only the missing questions needed to choose a causal design.
5. Do not run causal estimates until core variables and timing are clear.

## Cleaning actions to suggest

- Rename columns with spaces, punctuation, or non-identifier characters before formula-based OLS/RDD.
- Convert numeric-looking text columns with `pd.to_numeric(..., errors="coerce")`.
- Convert date-looking text columns with `pd.to_datetime(..., errors="coerce")`.
- Drop constant or all-missing variables from covariate matrices.
- Handle missing values explicitly. For core variables, prefer clear sample restrictions plus sensitivity checks.
- Encode categorical controls with `pd.get_dummies(..., drop_first=True)` before passing them to bundled estimators.
- Validate binary treatment coding as 0/1 before using OLS, PS, DID, or RDD treatment variables.
- Inspect extreme numeric values before deleting them; consider winsorization, logs, or influence diagnostics.
- For panel data, check duplicate unit-time pairs and sort by unit/time before DID or event-study work.

## Causal-design cautions

- Treatment must be measured before the outcome.
- Controls should be pre-treatment; avoid mediators, colliders, or variables mechanically affected by treatment.
- Missingness caused by treatment can change the target population.
- Trimming or matching changes the estimand; state the post-trimming target population.
- If automatic role candidates conflict with the user's design, trust the design only after checking variable definitions.
