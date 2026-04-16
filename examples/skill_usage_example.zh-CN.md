# Skill 使用示例：DID 政策评估

这个示例展示如何调用 `$econometrics`，让它不只是跑一个回归，而是按高级应用计量研究流程完成一份可用于论文初稿或研究备忘录的分析。

## 用户请求示例

```text
请使用 $econometrics 帮我做一份高级应用计量分析。

研究问题：最低工资上调是否影响城市就业率？
数据文件：/path/to/city_min_wage_panel.xlsx
数据结构：城市-年份面板数据，2008-2022 年。

变量：
- outcome: employment_rate，城市就业率，单位是百分点
- treatment: min_wage_reform，某城市某年是否已经执行最低工资上调政策，0/1
- entity id: city_id
- time: year
- controls: gdp_per_capita, industrial_share, population_log, education_share

我希望你按高级应用模式输出：
1. 明确 estimand
2. 写识别策略备忘录
3. 给出首选 DID / event-study 设定
4. 做平行趋势和事件研究诊断
5. 给出稳健性检验表
6. 解释结果时区分统计显著性、经济显著性和识别威胁
```

## Skill 应该如何处理

### 1. 进入高级应用模式

因为用户明确要求”高级应用计量分析”，应读取 `references/advanced_applied_workflow.md`，按高级应用模式组织输出，而不是只调用一个 DID 函数。

### 2. 先写 estimand

示例输出：

```text
目标 estimand 是最低工资上调对已执行改革城市就业率的 ATT。时间维度上，
主估计关注政策执行后的平均效应；事件研究进一步估计改革前后不同 event time
上的动态效应。目标总体是样本中在 2008-2022 年间发生最低工资上调的城市。
```

### 3. 写识别策略备忘录

示例输出：

```text
识别变异来自不同城市在不同年份执行最低工资上调的时点差异。朴素比较
treated 与 untreated 城市的就业率会有偏，因为改革城市可能在经济结构、
增长趋势、人口规模或产业组成上系统不同。DID 的核心假设是：在没有最低工资
上调的反事实世界中，处理城市和对照城市的就业率会沿着平行趋势演化。

这个假设不能被完全检验，但可以通过改革前 event-study leads、改革前 placebo
处理年份、城市固定效应、年份固定效应、以及时间变化控制变量来评估其可信度。
```

### 4. 数据检查与调用代码

示例代码骨架：

```python
import sys
from pathlib import Path
import pandas as pd

skill_dir = Path("/path/to/econometrics-skill")
sys.path.insert(0, str(skill_dir / "lib"))

from data_preprocess import get_column_info, load_table
from econometric_algorithm import (
    Staggered_Diff_in_Diff_regression,
    Staggered_Diff_in_Diff_Event_Study_regression,
    Staggered_Diff_in_Diff_Event_Study_visualization,
)

df = load_table("/path/to/city_min_wage_panel.xlsx", sheet_name=0)
print(get_column_info(df))

needed = [
    "city_id", "year", "employment_rate", "min_wage_reform",
    "gdp_per_capita", "industrial_share", "population_log", "education_share",
]
df = df[needed].dropna()
df["min_wage_reform"] = df["min_wage_reform"].astype(int)
df = df.sort_values(["city_id", "year"]).set_index(["city_id", "year"])

y = df["employment_rate"]
treat = df["min_wage_reform"]
x = df[["gdp_per_capita", "industrial_share", "population_log", "education_share"]]

did = Staggered_Diff_in_Diff_regression(
    dependent_variable=y,
    entity_treatment_dummy=treat,
    covariate_variables=x,
    entity_effect=True,
    time_effect=True,
    cov_type="cluster_entity",
)

event = Staggered_Diff_in_Diff_Event_Study_regression(
    dependent_variable=y,
    entity_treatment_dummy=treat,
    covariate_variables=x,
    see_back_length=4,
    see_forward_length=4,
    entity_effect=True,
    time_effect=True,
    cov_type="cluster_entity",
)

fig = Staggered_Diff_in_Diff_Event_Study_visualization(event, 4, 4)
fig.savefig("min_wage_event_study.png", dpi=150, bbox_inches="tight")
```

### 5. 稳健性检验矩阵

示例输出：

| 维度 | 主设定 | 稳健性版本 | 失败信号 |
|---|---|---|---|
| 控制变量 | 城市 FE、年份 FE、经济控制变量 | 去掉控制变量、加入省份线性趋势 | 系数符号或量级大幅变化 |
| 样本 | 全部城市-年份 | 去掉直辖市、只保留平衡面板 | 效果只由少数城市驱动 |
| 推断 | 按城市聚类 | 按省份聚类、wild bootstrap（若聚类数少） | 显著性只在便利标准误下存在 |
| 时间窗口 | 2008-2022 | 2010-2020、排除改革前一年 | 政策前已经出现“效果” |
| 事件研究 | leads/lags = 4 | leads/lags = 3 或 5 | leads 系统性偏离 0 |
| 安慰剂 | 无 | 随机提前政策年份、使用不应受影响的结果变量 | placebo 效果接近主效应 |

### 6. 结论写法

示例输出：

```text
首选 staggered DID 设定估计的是最低工资上调对改革城市就业率的平均处理效应。
如果 treatment_entity_treated 的系数为 -0.42，含义是：在城市固定效应、年份
固定效应和时间变化控制变量下，最低工资上调后，就业率平均下降 0.42 个百分点。

是否能解释为因果效应，取决于平行趋势假设。若事件研究中 Lead_D4+、Lead_D3、
Lead_D2 均接近 0 且不显著，这支持处理前趋势相似；若这些 lead 系数已经显著
偏离 0，则主 DID 估计不应被解释为干净的因果效应。

即使主系数显著，也需要报告经济显著性：0.42 个百分点相对于样本平均就业率
是多少？是否足以构成政策上重要的变化？最后还要说明剩余威胁，例如城市可能
基于未观测的劳动市场压力选择上调最低工资，或政策同时伴随其他劳动监管变化。
```

## 关键点

- `$econometrics` 不应只返回回归表。
- 高级模式必须先定义 estimand，再讨论识别策略。
- DID 的重点是平行趋势和处理时点，而不是交互项本身。
- 结果报告必须同时覆盖点估计、不确定性、经济意义、诊断证据和剩余识别威胁。
