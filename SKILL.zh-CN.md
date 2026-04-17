---
name: econometrics
description: >-
  面向表格数据的因果推断与应用计量分析，从上传数据诊断、清洗建议、
  快速处理效应估计到发表级应用研究设计。
  用于估计政策影响、ATE/ATT/LATE/ITT，或用 OLS、倾向得分、IV/2SLS、
  DID/事件研究、RDD、稳健性检验、证伪检验、识别策略备忘录等方式回答
  “X 对 Y 的影响是什么”。也用于中文请求：因果推断、政策评估、工具变量、
  双重差分、断点回归、平行趋势、内生性、选择偏误、数据清洗建议、
  稳健性检验、异质性分析，
  或“帮我做计量分析”。
metadata:
  short-description: "面向表格数据的因果推断"
  version: "1.3.0"
  author: "econometrics-agent"
---

# 计量经济学 Skill

这个 skill 为 Codex、Claude Code、OpenClaw 等 AI 编程助手提供一套经过整理的 **17 个因果推断估计器**（位于 `lib/econometric_algorithm.py`），并提供围绕用户识别策略选择合适方法的判断框架。对于进阶研究项目、审稿回复或发表级分析，它还提供更高级的应用计量研究工作流。

## 这个 skill 适合做什么

用于表格数据上的因果推断，也就是回答 **“在控制混杂因素后，处理 T 对结果 Y 的影响是什么？”** 这类问题。它使用的是应用经济学和社会科学中相对可辩护的方法。

这个 skill **不适合**纯预测、时间序列 ARIMA 预测，或机器学习模型训练。如果用户要的是预测准确率，而不是无偏的因果估计，应当引导他们换用更合适的工具，不要硬套这些方法。

## 运行环境要求

使用 Python 3.10+，并安装 `numpy`、`pandas`、`matplotlib`、`statsmodels`、`linearmodels`、`scipy`。如果要读取 `.xlsx/.xlsm` Excel 文件，还需要 `openpyxl`；如果要读取旧版 `.xls` Excel 文件，还需要 `xlrd`。

## 深度模式

- **快速模式**：当用户需要可信的一阶估计、探索性因果分析或方法选择时，使用下面的核心工作流。
- **高级应用模式**：当用户要求发表级分析、审稿级稳健性、识别策略批判、异质性分析、证伪检验或研究设计备忘录时，读取 `references/advanced_applied_workflow.md`。
- **项目工作流模式**：当用户需要从问题到最终报告的完整实证项目计划时，读取 `references/applied_project_workflow.md`。
- **数据诊断模式**：当用户上传数据或询问如何清洗数据时，使用 `lib/data_preprocess.py` 和 `references/data_preprocessing_advice.md`。
- **结果表模式**：当用户需要紧凑的模型对比表时，使用 `lib/result_tables.py` 和 `references/result_tables.md`。
- **诊断清单模式**：在把估计结果表述为因果证据前，读取 `references/diagnostic_checklist.md`。

## 核心工作流

每次分析都遵循同样的三步结构。不要跳过任何一步，因为跳过识别策略会让估计结果失去意义。

### 第 1 步：理解识别策略

在写任何代码之前，先和用户确认下面四个问题。如果用户没有说明，就主动询问：

1. **结果变量 Y 是什么？**（连续变量 / 二元变量 / 计数变量）
2. **处理变量 T 是什么？**（二元处理 / 连续处理 / 政策虚拟变量）
3. **为什么朴素的 `Y ~ T` 回归会有偏？**（选择偏误？反向因果？遗漏变量？）
4. **识别变异来自哪里？**（随机化 / 条件独立 / 工具变量 / 政策时点 / cutoff 断点）

第 4 个问题的答案决定方法家族。完整决策指南见 `references/method_selection.md`；当识别策略不明显时，先读它。

**快速映射：**

| 识别变异来源 | 方法家族 | 函数 |
|---|---|---|
| 随机或近似随机 | 带控制变量的 OLS | `ordinary_least_square_regression` |
| 基于可观测变量的选择 | 倾向得分方法 | `propensity_score_construction`、重叠性可视化、PSM/IPW 主估计 |
| 外生工具变量 Z 只通过 T 影响 Y | IV / 2SLS | `IV_2SLS_regression`、`IV_2SLS_IV_setting_test` |
| 政策变化 + 面板数据（前后 × 处理/对照） | DID | `Static_Diff_in_Diff_regression`、`Staggered_Diff_in_Diff_*`（3 个函数） |
| running variable 上存在清晰阈值 | RDD | `Sharp_*`、`Fuzzy_*`（3 个函数） |

### 第 2 步：检查数据，再调用算法

用 `lib.data_preprocess` 中的 `load_table()` 加载数据集。只要用户上传数据或询问如何清洗数据，就先运行 `analyze_dataset()` 和 `format_dataset_report()`，用诊断报告识别可能的变量角色、缺失、重复、类型转换、极端值和面板结构，再选择估计器。完整流程见 `references/data_preprocessing_advice.md`。

完成数据诊断后，和用户确认列名，然后**直接调用算法函数**。这些都是普通 Python 函数，返回拟合模型或估计结果，不是 agent。

数据以 `pd.Series` / `pd.DataFrame` 传入：

- `dependent_variable` = `df["Y"]`
- `treatment_variable` = `df["T"]`
- `covariate_variables` = `df[["X1", "X2", ...]]`（DataFrame，或 `None`）

不同函数的返回值不同，可能是拟合模型对象（statsmodels / linearmodels）、标量 ATE、pd.Series（倾向得分）、dict summary，或 matplotlib Figure。非法 `target_type` 会抛出 `ValueError`；不确定时先查 `references/method_details.md`。

每个函数的精确签名、参数含义和最小代码片段见 `references/method_details.md`。本轮对话中第一次调用某个方法前，先读对应部分。

### 第 3 步：用普通语言解释结果

不要把 `summary()` 输出直接丢给用户就结束。需要翻译成：

- **点估计**：方向和大小是什么，用结果变量的单位解释
- **统计显著性**：p 值相对常用阈值如何，但不要迷信 p<0.05
- **实际显著性**：效应大小在业务或研究语境中是否有意义
- **限制与假设**：这里最脆弱的识别假设是什么

不同方法家族的报告模板见 `references/interpretation.md`（例如 DID 的平行趋势、IV 的排除限制、PSM 的共同支撑等）。

### 高级模式交付物

对于严肃应用研究，不要只估计一个系数。应当产出这些内容：

1. **Estimand 陈述**：ATE/ATT/LATE/ITT、目标总体、时间跨度和结果变量尺度。
2. **识别策略备忘录**：识别变异来自哪里、需要哪些假设、朴素比较为什么失败，以及什么证据会推翻这个设计。
3. **模型设定矩阵**：基准设定、首选设定、更丰富控制变量、固定效应、聚类选择、样本限制和替代函数形式。
4. **诊断检验**：PS 的重叠性/平衡性，IV 的第一阶段/简约式，DID 的处理前趋势，RDD 的密度/协变量连续性。
5. **稳健性组合**：安慰剂结果、安慰剂处理时点/cutoff、带宽或 trimming 敏感性、聚类选择、影响点检查。
6. **异质性计划**：预先指定的子组或交互项、多重检验提醒，以及效应是否只能解释为探索性。
7. **研究级 caveats**：仍不可检验的假设是什么，现有证据如何支持它，以及更强的研究设计会是什么。

如需完整项目脚手架，读取 `references/applied_project_workflow.md`。如需分方法诊断，读取 `references/diagnostic_checklist.md`。如需模型对比结果表，使用 `lib/result_tables.py` 并参考 `references/result_tables.md`。

## 数据预处理（调用任何算法之前）

所有库函数都要求干净的数值型数据。对上传数据，先运行：

```python
from data_preprocess import analyze_dataset, format_dataset_report

analysis = analyze_dataset("data.xlsx", sheet_name=0)
print(format_dataset_report(analysis))
```

然后在加载数据后、调用估计器前做下面检查；跳过这些步骤是 cryptic error 的最常见来源：

1. **缺失值**：传入前先删除或填补 NaN。库函数会调用 `.astype(float)`，对象列中含 NaN 时可能报错。用 `df.isnull().sum()` 检查并显式处理。

2. **列名清理**：RDD 函数使用 statsmodels formula API（`smf.wls(formula, ...)`），列名中含空格、括号、短横线或其他特殊字符会失败。使用前把列名改成干净标识符：`df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)`。

3. **处理变量验证**：多数函数要求二元 0/1 处理虚拟变量。用 `df["T"].unique()` 检查；1/2、True/False、`"treated"`/`"control"` 等值会报错或被静默误解。显式转换：`df["T"] = (df["T"] == "treated").astype(int)`。

4. **Series 的 `.name` 属性**：RDD 和 OLS 函数使用 `series.name` 给输出系数命名。如果 Series 没有名字（例如从计算列生成），要显式设置：`treatment.name = "scholarship"`。

5. **DID 的面板数据 MultiIndex**：所有 DID 函数都要求 `(entity, time)` MultiIndex。调用前设置：`df = df.set_index(["firm_id", "year"])`。entity 和 time 层级必须可排序。

6. **类别变量编码**：作为协变量传入前，类别/字符串列必须先做 dummy 编码。使用 `pd.get_dummies(df[["industry"]], drop_first=True)`，再和数值协变量拼接。

## 运行代码

算法位于 `<skill_dir>/lib/econometric_algorithm.py`。从工作目录运行时，把 skill 的 `lib/` 加入 `sys.path`，或使用完整路径导入。可复用模式如下：

```python
import sys, os
SKILL_LIB = os.path.join(os.path.dirname(__file__), "lib")  # adjust to skill path
sys.path.insert(0, SKILL_LIB)

import pandas as pd
from econometric_algorithm import (
    ordinary_least_square_regression,
    propensity_score_construction,
    propensity_score_inverse_probability_weighting,
    IV_2SLS_regression,
    Static_Diff_in_Diff_regression,
    Sharp_Regression_Discontinuity_Design_regression,
)
from data_preprocess import analyze_dataset, format_dataset_report, get_column_info, load_table

df = load_table("data.xlsx", sheet_name=0)  # 也支持 .csv、.tsv、.xls、.xlsm
print(get_column_info(df))
analysis = analyze_dataset(df)
print(format_dataset_report(analysis))
```

运行代码时：

- 模块内部已经设置 `matplotlib.use("Agg")`，所以生成图像不需要显示环境
- 保存图像用 `fig.savefig("out.png", dpi=150, bbox_inches="tight")`，不要尝试 show
- 面板方法（DID）要求 DataFrame 有 **(entity, time) MultiIndex**，见 `method_details.md` 的 DID 部分

## 协方差 / 标准误

**OLS/RDD/IV 家族和 DID 家族使用完全不同的参数空间。** 混用会触发 RuntimeError。传入 `cov_type` 前先确认函数底层使用哪个库。

### OLS、RDD、IV、PS-regression、IPW-RA（基于 statsmodels）

参数名：`cov_info`（某些 PS 函数使用 `cov_type`）。

| 用户表述 | 传入值 |
|---|---|
| “robust” / “heteroskedasticity-robust” / “White” | `"HC1"` |
| “classical” / “default” / 未说明 | `"nonrobust"` |
| “clustered by firm” | `{"cluster": df["firm_id"]}` |
| “Newey-West, 4 lags” / HAC | `{"HAC": 4}` |

### DID 函数（基于 linearmodels PanelOLS）

参数名：`cov_type`。**字符串值完全不同**，不要传 `"HC1"` 或 dict。

| 用户表述 | 传入值 |
|---|---|
| “classical” / “default” / 未说明 | `"unadjusted"` |
| “robust” / “heteroskedasticity-robust” | `"robust"` |
| “clustered by entity / firm / individual” | `"cluster_entity"` |
| “clustered by time / year / period” | `"cluster_time"` |
| “two-way clustered” | `"cluster_both"` |

如果用户说得含糊（例如“面板数据用 robust errors”），DID 默认选择 `"cluster_entity"`，这是应用经济学中常见的默认做法（Bertrand, Duflo & Mullainathan 2004）。

## 常见坑（需要主动处理）

1. **PSM 没有检查共同支撑**：匹配前总是先运行 `propensity_score_visualize_propensity_score_distribution`；如果重叠性差，提醒用户并建议 trimming。
2. **DID 没有检查平行趋势**：对 staggered DID，运行事件研究（`Staggered_Diff_in_Diff_Event_Study_regression`）并查看处理前系数；它们应当在 0 附近较平坦。
3. **IV 使用弱工具变量**：总是调用 `IV_2SLS_IV_setting_test` 并报告控制协变量后的 partial first-stage F 统计量；经验规则是 F > 10。
4. **RDD 带宽选择错误**：默认带宽会主导结果。至少提供两个带宽并做敏感性检查。
5. **Fuzzy RDD 只返回裸比率**：默认使用 `target_type="summary"`，这样结果会包含 Wald LATE、第一阶段跳跃、近似 SE/CI、带宽内样本量和弱一阶段标记。只有兼容旧标量代码时才使用 `target_type="estimator"`。
6. **二元结果变量用 OLS**：线性概率模型可用于 ATE，但要提醒预测概率可能落在 [0, 1] 之外，并可建议 Logit/Probit 作为敏感性检查。

## 已知库限制（相关时要提前告诉用户）

下面是 `lib/econometric_algorithm.py` 实现本身的问题，无法通过不同调用方式修复。相关时要披露，让用户判断结果可信度。

### IV 2SLS 标准误向下偏

库里的 2SLS 是用两次 OLS 手写实现的。第二阶段 OLS 使用预测得到的 T-hat 残差计算标准误，而不是实际 T 的残差；没有应用正确的 2SLS 标准误调整（Wooldridge Ch. 15）。**报告的 p 值和置信区间会过于乐观。** 如果要做可靠的 IV 分析，建议直接使用 `linearmodels.iv.IV2SLS`，或用 R 的 `ivreg` 做交叉验证。

### IV 诊断不能证明排除限制

`IV_2SLS_IV_setting_test` 现在会报告包含协变量的第一阶段、简约式，以及一个残差伪检验。不要把残差伪检验称为排除限制检验。刚好识别的 IV 设计中，排除限制无法被直接检验；必须依靠制度背景、处理前协变量平衡、安慰剂结果和敏感性讨论来支撑。

### AIPW 不是双重稳健

当前实现只为控制组构造反事实（为 controls 预测 Y(1)），没有为处理组构造反事实（为 treated 预测 Y(0)）。标准 AIPW（Robins-Rotnitzky-Zhao 1994）需要两个方向。当前结果是不对称的，**实际上不具备双重稳健性**。主估计优先使用 PSM + IPW；如果需要真正的双重稳健估计，使用 `econml.dr.DRLearner` 或 `dowhy`。

### IPW-RA 对 IPW 权重取了平方根

IPW-RA 实现在加权回归前执行 `IPW = IPW ** 0.5`。这在 DR 文献中没有标准理论依据（Bang & Robins 2005）。报告结果时要说明这是非标准实现。

### 事件研究 lead/lag 赋值依赖 DataFrame 整数索引

代码使用 `each_index - policy_time_index`，这里是 pandas 的整数索引，不是真实时间距离。只有在面板平衡、按时间排序、且整数索引没有缺口时才有效。筛选过或非平衡面板可能得到错误的 lead/lag 赋值。调用事件研究函数前，务必保证面板平衡并已排序。

## 什么时候跳出内置方法

这 17 个函数覆盖常见场景，但有些请求需要自定义代码，例如 synthetic control、triple-diff、分位数处理效应、基于机器学习的异质性效应（Causal Forest、DML）。如果用户要求这些方法，坦诚说明这个 skill 没有覆盖，并主动提出用 statsmodels / linearmodels / scikit-learn 直接从头实现。

## 参考文件

- `references/method_selection.md`：选择合适估计器的决策指南
- `references/applied_project_workflow.md`：从研究问题到最终报告的完整实证项目工作流
- `references/advanced_applied_workflow.md`：高级应用工作流：estimand、识别备忘录、诊断、稳健性、异质性
- `references/data_preprocessing_advice.md`：上传数据的自动诊断和清洗建议工作流
- `references/diagnostic_checklist.md`：把估计结果表述为因果证据前的分方法检查清单
- `references/result_tables.md`：使用 `lib/result_tables.py` 生成紧凑模型对比表
- `references/method_details.md`：每个函数的精确签名和最小代码示例
- `references/interpretation.md`：每个方法家族的结果报告方式
- `examples/`：各方法家族可运行的端到端示例
