# Econometrics Skill

**Causal inference for tabular data** — 17 estimators from OLS to RDD, packaged as an AI coding-agent skill.

[中文说明](#中文说明)

## What this is

A skill (structured prompt + bundled library) that gives AI coding agents the ability to run defensible causal-inference analyses on user-provided datasets. It covers the standard applied-econometrics toolkit:

| Method family | Functions | Estimand |
|---|---|---|
| OLS with controls | 1 | ATE (under CIA) |
| Propensity Score (PS, PSM, IPW, AIPW, IPW-RA) | 7 | ATE / ATT |
| IV / 2SLS | 2 | LATE |
| Difference-in-Differences (static, staggered, event study) | 4 | ATT / dynamic effects |
| Regression Discontinuity (sharp, fuzzy, global polynomial) | 3 | LATE at cutoff |

The skill includes method selection guidance, data preprocessing checklists, covariance/SE mapping tables, known library caveats, interpretation templates, an end-to-end project workflow, diagnostic checklists, result-table helpers, and an advanced applied workflow.

## Supported agent frameworks

- **Claude Code** / **OpenClaw** — reads `SKILL.md` directly
- **OpenAI Agents** — see `agents/openai.yaml`
- Any framework that can read a Markdown skill definition

## Quick start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run an example
cd examples
python ols_example.py
```

## Project structure

```
econometrics-skill/
├── SKILL.md                              # Main skill definition (English)
├── SKILL.zh-CN.md                        # 中文版 skill 定义
├── lib/
│   ├── econometric_algorithm.py          # 17 causal-inference functions
│   ├── data_preprocess.py                # load_table + get_column_info
│   └── result_tables.py                  # compact model comparison tables
├── references/
│   ├── method_selection.md               # Decision tree for picking the right estimator
│   ├── method_details.md                 # Exact signatures and minimal code per function
│   ├── interpretation.md                 # Reporting templates per method family
│   ├── applied_project_workflow.md       # End-to-end empirical project workflow
│   ├── diagnostic_checklist.md           # Method-specific diagnostic checklist
│   ├── result_tables.md                  # Result table helper guide
│   └── advanced_applied_workflow.md      # Advanced applied workflow
├── examples/
│   ├── ols_example.py                    # OLS: schooling → wages
│   ├── psm_example.py                    # PSM + IPW with bootstrap
│   ├── iv_example.py                     # IV/2SLS: Card-style college → earnings
│   ├── did_example.py                    # Static DID: minimum wage panel
│   ├── did_event_study_example.py        # Staggered DID + event study
│   ├── rdd_example.py                    # Sharp RDD: merit scholarship
│   ├── rdd_fuzzy_example.py              # Fuzzy RDD: housing subsidy
│   └── skill_usage_example.zh-CN.md      # 中文使用示例：DID 政策评估
├── agents/
│   └── openai.yaml                       # OpenAI agent interface config
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies

- Python 3.10+
- numpy, pandas, matplotlib, statsmodels, linearmodels, scipy, openpyxl

## Practical helpers

- `references/applied_project_workflow.md` guides users from question to final report.
- `references/diagnostic_checklist.md` gives method-specific checks before causal interpretation.
- `lib/result_tables.py` builds compact model comparison tables for statsmodels and linearmodels results.

## Known limitations

The library functions have several documented caveats (IV SE bias, non-standard AIPW, event study viz bugs, etc.). These are fully documented in `SKILL.md` § "Known library limitations" and in `references/method_details.md`. The skill instructs the agent to disclose these to users when relevant.

## License

MIT

---

## 中文说明

**面向表格数据的因果推断** — 17 个估计器，从 OLS 到 RDD，打包为 AI 编程助手的 skill。

### 这是什么

一个结构化的 skill（提示词 + 算法库），让 AI 编程助手能够在用户提供的数据集上运行可辩护的因果推断分析。涵盖标准应用计量工具箱：

- **OLS**：带控制变量的回归
- **倾向得分方法**：PS 构建、可视化、匹配（PSM）、逆概率加权（IPW）、PS 回归、AIPW、IPW-RA
- **工具变量**：IV / 2SLS + 诊断检验
- **双重差分**：静态 DID、交错 DID、事件研究 + 可视化
- **断点回归**：Sharp RDD、Fuzzy RDD（两步法 + 全局多项式）

### 快速开始

```bash
pip install -r requirements.txt
cd examples
python ols_example.py
```

### 深度模式

- **快速模式**：核心三步工作流（识别策略 → 数据检查 → 结果解释）
- **项目工作流模式**：研究问题 → 变量表 → 识别策略 → 诊断 → 稳健性 → 最终报告
- **结果表模式**：整理多个模型的系数、标准误、显著性标记和拟合统计量
- **诊断清单模式**：在因果解释前检查各方法的关键风险
- **高级应用模式**：高级研究流程（estimand → 识别备忘录 → 诊断 → 稳健性 → 异质性）

详细中文说明见 `SKILL.zh-CN.md`，使用示例见 `examples/skill_usage_example.zh-CN.md`。
