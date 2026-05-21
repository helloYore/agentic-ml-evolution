---
name: kaggle
description: "Kaggle competition automation — multi-round iterative optimization for tabular ML competitions. Download data, build models, submit, iterate via a state machine loop, and track evolution in an external report for context-safe persistence."
category: data-science
version: 1.0.0
author: Hermes Agent
tags: [kaggle, competition, ml, tabular, iterative-optimization, ensemble]
---

# Kaggle Competition Automation

Standard operating procedure for Kaggle tabular ML competitions. Follow the state machine strictly, maintain an external evolution report to survive context compression, and iterate toward a better Public Score.

**Prerequisites**: `kaggle` CLI (`pip install kaggle` + API key at ~/.kaggle/kaggle.json), scikit-learn, pandas, numpy, and any model library used (xgboost, lightgbm, catboost).

---

## Quick Start

```bash
# Download a competition
kaggle competitions download -c <competition-name>
unzip -o "<competition-name>.zip"

# Inspect data
python3 -c "
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f'Train: {train.shape}, Test: {test.shape}')
print(f'Missing: train={train.isnull().sum().sum()}, test={test.isnull().sum().sum()}')
print(f'Columns with NAs: {list(train.columns[train.isnull().any()])}')
"
```

---

## State Machine (strict order)

### State 1 — Data & Baseline (N=1 only)
- Download + unzip competition data
- Quick data inspection: shape, missing values, target distribution, numeric vs categorical
- Write a **simple baseline** (RandomForest, median imputation, OneHot encoding)
- Save as `train.py`
- Generate `submission.csv` and submit with tag: `"Iteration 1 Baseline"`

### State 2 — Feedback & Record (every iteration)
- `sleep 60` — Kaggle scoring is async, must block
- Query Public Score: `kaggle competitions submissions -c <name> -v`
- Append to **evolution_report.md** with format:

```markdown
## Iteration N — <strategy-summary>

### 策略
<bullet points: what changed>

### 结果
- CV R²: <value>
- Public Score (RMSLE): **<value>**

### 缺陷分析
1. <issue 1>
2. <issue 2>
```

### State 3 — Reflection & Evolution (N < max_rounds)
- Read `evolution_report.md` to confirm previous records (avoids duplication)
- Propose **2-3 specific optimization hypotheses** based on prior analysis
- Append hypotheses to `evolution_report.md`
- Rewrite `train.py` with the new strategy
- Generate new `submission.csv` and submit
- Increment N, loop back to State 2

### State 4 — Final Synthesis (N = max_rounds)
- Read the full `evolution_report.md`
- Append a **global summary** section: score trajectory, quantified impact of each strategy, lessons learned
- Print `"N轮迭代闭环结束"`

---

## Evolution Report — External Memory Pattern

The `evolution_report.md` serves as **context-safe external memory**. All analysis, decision logic, and experimental results go here — NOT in memory tools or the agent's internal context.

```markdown
# <Competition Name> — Evolution Report

---

## Iteration 0 — 初始化
开始时间: <datetime>
目标: <N>轮自动化迭代优化
当前目录: <pwd>
比赛: <competition-name>

---
```

**Rules**:
- Read the report *before* writing any new code iteration to confirm what was already tried
- Append, never overwrite prior iterations
- Include CV score AND Public Score for every iteration

---

## Real-World Validation: House Prices 10-Round Evolution

2026-05-03 实战记录 — 比赛: House Prices, 10 rounds, DS-V4-Flash

| Round | Strategy | Public Score (RMSLE) | Δ vs Baseline |
|-------|----------|----------------------|---------------|
| 1 | RandomForest baseline | 0.14640 | — |
| 3 | XGBoost + feature engineering | **0.12975** ⭐ | -11.4% |
| 6 | ElasticNetCV + StandardScaler | **0.12875** ⭐ | -12.1% |
| 7 | XGB + ElasticNet ensemble | **0.12551** ⭐ | -14.3% |
| 10 | 3-model ensemble (final) | **0.12381** 🏆 | -15.4% |

Cost: ~¥1 RMB | Time: 20 min | Result: Top ~20%

---

## Common Model Strategies (ordered by typical impact)

| Iteration | Strategy | Expected Impact |
|-----------|----------|-----------------|
| 1 | RandomForest / baseline | Establish baseline |
| 2 | log1p target, outlier removal, skew correction | Small (±2-5%) |
| 3 | XGBoost + feature engineering (TotalSF, TotalBath, quality interactions) | **Large (10-15%)** |
| 4 | LightGBM / CatBoost for native categorical handling | Marginal (+/- 2%) |
| 5 | Hyperparameter tuning | Small (0-5%) |
| 6 | ElasticNet / Ridge / Lasso + StandardScaler | **Large (5-10%)** after log target |
| 7+ | Ensemble (weighted avg), stacking | Small-medium (2-5%) |

---

## Feature Engineering Template

```python
# Area features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath']

# Quality interactions
all_data['OverallQual_SF'] = all_data['OverallQual'] * all_data['TotalSF']

# Age features
all_data['HouseAge'] = 2026 - all_data['YearBuilt']

# Ordinal encoding
quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

# Log transform skewed numerics
all_data[col+'_log'] = np.log1p(np.maximum(all_data[col].fillna(0), 0))

# Neighborhood target encoding
nb_mean = y_raw.groupby(train['Neighborhood']).mean().to_dict()
```

---

## Pitfalls

1. **Over-featurization**: Too many features + OneHot = dimension explosion. Less is often more.
2. **LightGBM ≠ always better**: XGBoost (level-wise) outperforms LightGBM (leaf-wise) on small datasets (~1500 rows).
3. **Hyperparameter over-regularization**: Too much reg_alpha/lambda cripples XGBoost. Validate with CV.
4. **CatBoost dtype**: cat_features must be strings or integers — convert with `.astype(str)` after filling NAs.
5. **CV vs Public Score**: They don't always correlate. Trust Public Score for final decisions.
6. **Kaggle submission delay**: Always `sleep 60` before querying scores.

---

## Related Skills
- `test-driven-development` — similar RED-GREEN-REFACTOR loop, adapted for ML
- `systematic-debugging` — root cause analysis for model regression
