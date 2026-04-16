# Result Tables

Use this reference when the user needs compact model comparison tables. The helper in `lib/result_tables.py` supports common statsmodels and linearmodels fitted results.

## Basic usage

```python
import sys
from pathlib import Path

skill_dir = Path("/path/to/econometrics-skill")
sys.path.insert(0, str(skill_dir / "lib"))

from result_tables import regression_table, to_markdown_table

table = regression_table(
    models=[m1, m2, m3],
    model_names=["Baseline", "Controls", "FE"],
    terms=["T", "X1"],
    digits=3,
)

print(to_markdown_table(table))
```

## Export options

```python
# Markdown string
markdown = to_markdown_table(table)

# CSV
table.to_csv("regression_table.csv")

# LaTeX, if the environment supports it
table.to_latex("regression_table.tex")
```

## Reporting rules

- Always name the estimand and sample above the table.
- Include N and R2 where available.
- Explain fixed effects, controls, and clustering in notes.
- Do not hide diagnostics in the table. Report them next to the table.
- Do not treat stars as a substitute for effect size or design credibility.

## Suggested table note

```text
Notes: Standard errors are in parentheses. *** p<0.01, ** p<0.05, * p<0.10.
All specifications use the sample described in the text. The preferred
specification is chosen based on the identification strategy, not on the
smallest p-value.
```
